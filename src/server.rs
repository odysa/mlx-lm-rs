use std::{
    collections::{BTreeSet, HashMap},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::mpsc as std_mpsc,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, Context, Result};
use tokio::sync::mpsc;
use tokio::task::JoinHandle as TokioJoinHandle;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::utility::{
    UtilityCallId, UtilityOutput, UtilityResultEnvelope,
};
use vllm_engine_core_client::protocol::{
    encode_msgpack, EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest,
    EngineCoreRequestType, EngineCoreSamplingParams, ModelDtype, StopReason,
};
use vllm_engine_core_client::{EngineId, TransportMode};
use vllm_server::{
    ChatTemplateContentFormatOption, Config, CoordinatorMode, HttpListenerMode, ParserSelection,
    RendererSelection,
};
use zeromq::{
    prelude::{Socket, SocketRecv, SocketSend},
    util::PeerIdentity,
    DealerSocket, PushSocket, SocketOptions, ZmqMessage,
};

use crate::{
    config::load_config,
    generate::Generator,
    loader::{list_weight_files, resolve_model_dir},
    models::qwen3::Model,
};

const ENGINE_INDEX: u32 = 0;

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub model: String,
    pub host: String,
    pub port: u16,
    pub prefill_step_size: NonZeroUsize,
}

#[derive(Debug)]
struct GenerateRequest {
    prompt_tokens: Vec<u32>,
    temperature: f32,
    ignore_eos: bool,
    max_tokens: usize,
    token_tx: mpsc::UnboundedSender<TokenEvent>,
}

#[derive(Debug)]
enum TokenEvent {
    Token(u32),
    Finished { reason: EngineCoreFinishReason },
    Error { message: String },
}

struct EngineHandle {
    submit_tx: mpsc::UnboundedSender<GenerateRequest>,
}

impl EngineHandle {
    fn submit(&self, req: GenerateRequest) -> Result<()> {
        self.submit_tx
            .send(req)
            .map_err(|_| anyhow::anyhow!("model worker is not running"))
    }
}

pub async fn serve(cfg: ServeConfig) -> Result<()> {
    let model_dir = resolve_model_dir(&cfg.model).context("resolving model dir")?;
    eprintln!("[mlx-lm-rs] using model dir: {}", model_dir.display());

    let model_cfg = load_config(&model_dir).context("loading config.json")?;
    let max_model_len = u32::try_from(model_cfg.max_position_embeddings)
        .ok()
        .filter(|len| *len > 0)
        .unwrap_or(4096);
    let handle = spawn_model_worker(model_dir.clone(), cfg.prefill_step_size)?;

    let shutdown = shutdown_token_from_ctrl_c();
    serve_vllm_frontend(
        handle,
        cfg.model,
        model_dir,
        cfg.host,
        cfg.port,
        max_model_len,
        shutdown,
    )
    .await
}

fn spawn_model_worker(model_dir: PathBuf, prefill_step_size: NonZeroUsize) -> Result<EngineHandle> {
    let (submit_tx, mut submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
    let (ready_tx, ready_rx) = std_mpsc::channel::<std::result::Result<(), String>>();

    let join_handle = thread::Builder::new()
        .name("mlx-lm-rs-model-worker".into())
        .spawn(move || match init_model_worker(&model_dir) {
            Ok((mut model, eos_ids)) => {
                let _ = ready_tx.send(Ok(()));
                while let Some(req) = submit_rx.blocking_recv() {
                    run_generation(&mut model, &eos_ids, prefill_step_size, req);
                }
            }
            Err(error) => {
                let message = format!("{error:#}");
                let _ = ready_tx.send(Err(message.clone()));
                eprintln!("[mlx-lm-rs] model worker failed to initialize: {message}");
            }
        })
        .context("spawning model worker")?;

    match ready_rx
        .recv()
        .context("waiting for model worker startup")?
    {
        Ok(()) => {
            drop(join_handle);
            Ok(EngineHandle { submit_tx })
        }
        Err(message) => anyhow::bail!("model worker failed to initialize: {message}"),
    }
}

fn init_model_worker(model_dir: &Path) -> Result<(Model, Vec<u32>)> {
    let model_cfg = load_config(model_dir).context("loading config.json")?;
    let eos_ids = model_cfg
        .eos_token_id
        .as_ref()
        .map(|x| x.ids())
        .unwrap_or_default();

    let mut model = Model::new(model_cfg).context("constructing model")?;
    let shards = list_weight_files(model_dir).context("listing weight files")?;
    eprintln!("[mlx-lm-rs] loading {} weight shard(s)...", shards.len());
    model.load_weights(&shards).context("loading weights")?;
    eprintln!("[mlx-lm-rs] weights loaded; accepting vLLM requests");

    Ok((model, eos_ids))
}

fn run_generation(
    model: &mut Model,
    eos_ids: &[u32],
    prefill_step_size: NonZeroUsize,
    req: GenerateRequest,
) {
    let mut completion_tokens = 0usize;
    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let request_eos_ids = if req.ignore_eos { &[] } else { eos_ids };
    let result = (|| -> crate::Result<EngineCoreFinishReason> {
        let mut generator = Generator::new(
            model,
            &req.prompt_tokens,
            max_tokens,
            temperature,
            request_eos_ids.to_vec(),
            prefill_step_size,
        )?;

        for token in &mut generator {
            let id = token?;
            completion_tokens += 1;
            if req.token_tx.send(TokenEvent::Token(id)).is_err() {
                return Ok(EngineCoreFinishReason::Error);
            }
        }

        Ok(if completion_tokens >= max_tokens && max_tokens > 0 {
            EngineCoreFinishReason::Length
        } else {
            EngineCoreFinishReason::Stop
        })
    })();

    match result {
        Ok(reason) => {
            let _ = req.token_tx.send(TokenEvent::Finished { reason });
        }
        Err(error) => {
            let _ = req.token_tx.send(TokenEvent::Error {
                message: format!("{error:#}"),
            });
        }
    }
}

struct LocalEngineBridge {
    input_address: String,
    output_address: String,
    handle: EngineHandle,
    max_model_len: u32,
}

impl LocalEngineBridge {
    async fn run(self, shutdown: CancellationToken) -> Result<()> {
        wait_for_ipc_endpoint(&self.input_address, &shutdown).await?;
        wait_for_ipc_endpoint(&self.output_address, &shutdown).await?;

        let engine_id = EngineId::from_engine_index(ENGINE_INDEX);
        let mut socket_options = SocketOptions::default();
        socket_options.peer_identity(PeerIdentity::try_from(engine_id)?);

        let mut input = DealerSocket::with_options(socket_options);
        input
            .connect(&self.input_address)
            .await
            .with_context(|| format!("connecting local engine input {}", self.input_address))?;

        let ready = EngineCoreReadyResponse {
            max_model_len: self.max_model_len as u64,
            num_gpu_blocks: 0,
            dp_stats_address: None,
            dtype: ModelDtype::BFloat16,
            vllm_version: "mlx-lm-rs-local-bridge".to_string(),
        };
        input
            .send(ZmqMessage::from(encode_msgpack(&ready)?))
            .await
            .context("sending local engine ready response")?;

        let mut output = PushSocket::new();
        output
            .connect(&self.output_address)
            .await
            .with_context(|| format!("connecting local engine output {}", self.output_address))?;

        let (output_tx, output_rx) = mpsc::unbounded_channel();
        let output_task = tokio::spawn(output_loop(output, output_rx));
        let (done_tx, mut done_rx) = mpsc::unbounded_channel::<String>();
        let mut active: HashMap<String, TokioJoinHandle<()>> = HashMap::new();

        eprintln!(
            "[mlx-lm-rs] local vLLM engine bridge connected: max_model_len={}",
            self.max_model_len
        );

        loop {
            tokio::select! {
                () = shutdown.cancelled() => break,
                Some(request_id) = done_rx.recv() => {
                    active.remove(&request_id);
                }
                recv = input.recv() => {
                    let message = recv.context("receiving local engine request")?;
                    if let Err(error) = self.handle_message(
                        message,
                        &output_tx,
                        &done_tx,
                        &mut active,
                    ) {
                        eprintln!("[mlx-lm-rs] local engine bridge request failed: {error:#}");
                    }
                }
            }
        }

        for (_, task) in active {
            task.abort();
        }
        drop(output_tx);
        output_task.abort();
        Ok(())
    }

    fn handle_message(
        &self,
        message: ZmqMessage,
        output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
        done_tx: &mpsc::UnboundedSender<String>,
        active: &mut HashMap<String, TokioJoinHandle<()>>,
    ) -> Result<()> {
        let frames = message.into_vec();
        if frames.len() != 2 {
            bail!(
                "expected 2 local engine request frames, got {}",
                frames.len()
            );
        }

        match frames[0].as_ref() {
            ty if ty == EngineCoreRequestType::Add.to_frame().as_ref() => {
                let request: EngineCoreRequest =
                    vllm_engine_core_client::protocol::decode_msgpack(&frames[1])?;
                self.start_request(request, output_tx, done_tx, active)
            }
            ty if ty == EngineCoreRequestType::Abort.to_frame().as_ref() => {
                let request_ids: Vec<String> =
                    vllm_engine_core_client::protocol::decode_msgpack(&frames[1])?;
                for request_id in request_ids {
                    if let Some(task) = active.remove(&request_id) {
                        task.abort();
                    }
                }
                Ok(())
            }
            ty if ty == EngineCoreRequestType::Utility.to_frame().as_ref() => {
                let (_client_index, call_id, method_name, _args): (
                    u32,
                    UtilityCallId,
                    String,
                    rmpv::Value,
                ) = rmp_serde::from_slice(&frames[1])?;
                send_utility_response(output_tx, call_id, &method_name)
            }
            other => bail!("unsupported local engine request type frame: {other:?}"),
        }
    }

    fn start_request(
        &self,
        request: EngineCoreRequest,
        output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
        done_tx: &mpsc::UnboundedSender<String>,
        active: &mut HashMap<String, TokioJoinHandle<()>>,
    ) -> Result<()> {
        let EngineCoreRequest {
            request_id,
            prompt_token_ids,
            sampling_params,
            ..
        } = request;
        let Some(prompt_tokens) = prompt_token_ids else {
            send_terminal_output(output_tx, request_id, EngineCoreFinishReason::Error, None)?;
            return Ok(());
        };
        let Some(sampling_params) = sampling_params else {
            send_terminal_output(output_tx, request_id, EngineCoreFinishReason::Error, None)?;
            return Ok(());
        };

        let max_tokens = sampling_params.max_tokens as usize;
        let temperature = sampling_temperature(&sampling_params);
        let ignore_eos = ignore_eos(&sampling_params);
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        self.handle.submit(GenerateRequest {
            prompt_tokens,
            temperature,
            ignore_eos,
            max_tokens,
            token_tx,
        })?;

        let output_tx = output_tx.clone();
        let done_tx = done_tx.clone();
        let task_request_id = request_id.clone();
        let task = tokio::spawn(async move {
            run_request_stream(task_request_id.clone(), token_rx, output_tx).await;
            let _ = done_tx.send(task_request_id);
        });
        active.insert(request_id, task);
        Ok(())
    }
}

async fn serve_vllm_frontend(
    handle: EngineHandle,
    model_id: String,
    model_dir: PathBuf,
    host: String,
    port: u16,
    max_model_len: u32,
    shutdown: CancellationToken,
) -> Result<()> {
    let namespace = local_ipc_namespace()?;
    let input_address = ipc_endpoint(&namespace, "input.sock");
    let output_address = ipc_endpoint(&namespace, "output.sock");

    let bridge = LocalEngineBridge {
        input_address: input_address.clone(),
        output_address: output_address.clone(),
        handle,
        max_model_len,
    };
    let bridge_shutdown = shutdown.child_token();
    let bridge_task = tokio::spawn(async move {
        if let Err(error) = bridge.run(bridge_shutdown).await {
            eprintln!("[mlx-lm-rs] local vLLM engine bridge exited: {error:#}");
        }
    });

    let served_model_name = vec![model_id.clone()];
    let config = Config {
        transport_mode: TransportMode::Bootstrapped {
            input_address,
            output_address,
            engine_count: 1,
            ready_timeout: Duration::from_secs(30),
        },
        coordinator_mode: CoordinatorMode::None,
        model: model_dir.to_string_lossy().into_owned(),
        served_model_name,
        listener_mode: HttpListenerMode::BindTcp { host, port },
        tool_call_parser: ParserSelection::default(),
        reasoning_parser: ParserSelection::default(),
        renderer: RendererSelection::default(),
        chat_template: None,
        default_chat_template_kwargs: None,
        chat_template_content_format: ChatTemplateContentFormatOption::default(),
        enable_log_requests: true,
        enable_request_id_headers: false,
        disable_log_stats: true,
        grpc_port: None,
        shutdown_timeout: Duration::from_secs(10),
    };

    let result = vllm_server::serve(config, shutdown).await;
    bridge_task.abort();
    let _ = std::fs::remove_dir_all(namespace);
    result
}

async fn run_request_stream(
    request_id: String,
    mut token_rx: mpsc::UnboundedReceiver<TokenEvent>,
    output_tx: mpsc::UnboundedSender<EngineCoreOutputs>,
) {
    while let Some(event) = token_rx.recv().await {
        match event {
            TokenEvent::Token(id) => {
                if send_token_output(&output_tx, &request_id, vec![id]).is_err() {
                    return;
                }
            }
            TokenEvent::Finished { reason } => {
                let _ = send_terminal_output(&output_tx, request_id, reason, None);
                return;
            }
            TokenEvent::Error { message } => {
                eprintln!("[mlx-lm-rs] request {request_id} failed: {message}");
                let _ = send_terminal_output(
                    &output_tx,
                    request_id,
                    EngineCoreFinishReason::Error,
                    Some(StopReason::Text(message)),
                );
                return;
            }
        }
    }
}

async fn output_loop(
    mut output: PushSocket,
    mut output_rx: mpsc::UnboundedReceiver<EngineCoreOutputs>,
) -> Result<()> {
    while let Some(outputs) = output_rx.recv().await {
        output
            .send(ZmqMessage::from(encode_msgpack(&outputs)?))
            .await
            .context("sending local engine output")?;
    }
    Ok(())
}

fn send_token_output(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    request_id: &str,
    token_ids: Vec<u32>,
) -> Result<()> {
    send_outputs(
        output_tx,
        EngineCoreOutputs {
            engine_index: ENGINE_INDEX,
            outputs: vec![engine_output(request_id.to_string(), token_ids, None, None)],
            timestamp: now_secs_f64(),
            ..Default::default()
        },
    )
}

fn send_terminal_output(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    request_id: String,
    finish_reason: EngineCoreFinishReason,
    stop_reason: Option<StopReason>,
) -> Result<()> {
    send_outputs(
        output_tx,
        EngineCoreOutputs {
            engine_index: ENGINE_INDEX,
            outputs: vec![engine_output(
                request_id.clone(),
                Vec::new(),
                Some(finish_reason),
                stop_reason,
                None,
            )],
            finished_requests: Some(BTreeSet::from([request_id])),
            timestamp: now_secs_f64(),
            ..Default::default()
        },
    )
}

fn send_utility_response(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    call_id: UtilityCallId,
    method_name: &str,
) -> Result<()> {
    let result = match method_name {
        "is_sleeping" | "reset_prefix_cache" => rmpv::ext::to_value(false)?,
        "sleep" | "wake_up" | "reset_mm_cache" | "reset_encoder_cache" | "collective_rpc" => {
            rmpv::Value::Nil
        }
        _ => rmpv::Value::Nil,
    };

    send_outputs(
        output_tx,
        EngineCoreOutputs {
            engine_index: ENGINE_INDEX,
            utility_output: Some(UtilityOutput {
                call_id,
                failure_message: None,
                result: Some(UtilityResultEnvelope::without_type_info(result)),
            }),
            timestamp: now_secs_f64(),
            ..Default::default()
        },
    )
}

fn send_outputs(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    outputs: EngineCoreOutputs,
) -> Result<()> {
    output_tx
        .send(outputs)
        .map_err(|_| anyhow::anyhow!("local engine output channel closed"))
}

fn engine_output(
    request_id: String,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
    stop_reason: Option<StopReason>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id,
        new_token_ids,
        new_logprobs: None,
        new_prompt_logprobs_tensors: None,
        pooling_output: None,
        finish_reason,
        stop_reason,
        events: None,
        kv_transfer_params: None,
        trace_headers: None,
        prefill_stats: None,
        routed_experts: None,
        num_nans_in_logits: 0,
    }
}

fn sampling_temperature(params: &EngineCoreSamplingParams) -> f32 {
    if params.temperature <= 0.0 {
        0.0
    } else {
        params.temperature
    }
}

fn ignore_eos(params: &EngineCoreSamplingParams) -> bool {
    params.eos_token_id.is_none() && params.stop_token_ids.is_empty()
}

fn now_secs_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn local_ipc_namespace() -> Result<PathBuf> {
    let base_dir =
        std::env::var_os("MLX_LM_RS_IPC_DIR").map_or_else(|| PathBuf::from("/tmp"), PathBuf::from);
    let uuid = uuid::Uuid::new_v4().to_string();
    let path = base_dir.join(format!("mlx-lm-rs-{}-{}", std::process::id(), &uuid[..8]));
    std::fs::create_dir_all(&path)
        .with_context(|| format!("creating IPC namespace {}", path.display()))?;
    Ok(path)
}

fn ipc_endpoint(namespace: &Path, name: &str) -> String {
    format!("ipc://{}", namespace.join(name).to_string_lossy())
}

async fn wait_for_ipc_endpoint(address: &str, shutdown: &CancellationToken) -> Result<()> {
    let Some(path) = address.strip_prefix("ipc://") else {
        return Ok(());
    };
    let path = Path::new(path);
    loop {
        if path.exists() {
            return Ok(());
        }
        tokio::select! {
            () = shutdown.cancelled() => bail!("shutdown before IPC endpoint appeared"),
            () = tokio::time::sleep(Duration::from_millis(20)) => {}
        }
    }
}

fn shutdown_token_from_ctrl_c() -> CancellationToken {
    let token = CancellationToken::new();
    let shutdown = token.clone();
    tokio::spawn(async move {
        if let Err(error) = tokio::signal::ctrl_c().await {
            eprintln!("[mlx-lm-rs] failed to install CTRL+C handler: {error}");
        }
        shutdown.cancel();
    });
    token
}

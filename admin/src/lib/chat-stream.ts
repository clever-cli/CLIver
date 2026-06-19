export interface ChatArtifact {
  path: string;
  media_type: string;
  size?: number;
}

export interface ChatStreamEvent {
  type: "thinking" | "text" | "tool" | "tool_use" | "tool_result" | "status" | "session" | "done" | "error";
  content?: string;
  text?: string;
  message?: string;
  data?: unknown;
  artifacts?: ChatArtifact[];
  session_id?: string;
  __llm_trace__?: Array<Record<string, unknown>>;
}

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  reasoning_content?: string;
}

export interface ChatStreamConfig {
  message: string;
  abortSignal?: AbortSignal;
  onEvent: (event: ChatStreamEvent) => void;
  onError: (error: Error) => void;
  onDone: (fullText: string, artifacts: ChatArtifact[], sessionId?: string, llmTrace?: Array<Record<string, unknown>>) => void;
  // Called as soon as the backend creates the session — before any content.
  // The UI can navigate to the conversation URL immediately.
  onSessionReady?: (sessionId: string) => void;

  agent?: string;
  model?: string;
  systemMessage?: string;
  conversationHistory?: ConversationMessage[];
  conversationId?: string;
  filterTools?: string[];
  saveMediaDir?: string;
  // Override the API endpoint (default: /admin/api/chat).
  apiPath?: string;
  // Extra fields to include in the request body.
  extraBody?: Record<string, unknown>;
}

export async function streamChat(config: ChatStreamConfig): Promise<void> {
  const { message, abortSignal, onEvent, onError, onDone } = config;

  const body = JSON.stringify({
    message,
    agent: config.agent,
    model: config.model,
    system_message: config.systemMessage,
    conversation_history: config.conversationHistory,
    session_id: config.conversationId,
    filter_tools: config.filterTools,
    save_media_dir: config.saveMediaDir,
    ...config.extraBody,
  });

  try {
    const apiPath = config.apiPath || "/admin/api/chat";
    const response = await fetch(apiPath, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      signal: abortSignal,
    });

    if (!response.ok) {
      const text = await response.text();
      onError(new Error(`Chat error ${response.status}: ${text}`));
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      onError(new Error("No response body"));
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";
    let artifacts: ChatArtifact[] = [];

    const normalizeEvent = (raw: Record<string, unknown>): ChatStreamEvent => {
      if (raw.type === "chunk" || raw.type === "content") {
        return { type: "text", content: raw.content as string };
      }
      return raw as unknown as ChatStreamEvent;
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const raw = JSON.parse(line.slice(6));
            const event = normalizeEvent(raw);

            onEvent(event);

            if (event.type === "session" && event.session_id) {
              config.onSessionReady?.(event.session_id);
              continue;
            }
            if (event.type === "text" && event.content) {
              fullText += event.content;
            }
            if (event.type === "done") {
              const rawArtifacts = (raw.media || raw.media_files) as ChatArtifact[] | undefined;
              if (rawArtifacts) {
                artifacts = rawArtifacts;
              }
              const sessionId = raw.session_id as string | undefined;
              const trace = raw.__llm_trace__ as Array<Record<string, unknown>> | undefined;
              onDone(fullText || (event.text || event.content as string) || "", artifacts, sessionId, trace);
              return;
            }
            if (event.type === "error") {
              onError(new Error(event.message || "Unknown error"));
              return;
            }
          } catch {
            // Skip unparseable lines
          }
        }
      }
    }

    onDone(fullText, artifacts);
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === "AbortError") {
      return;
    }
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}

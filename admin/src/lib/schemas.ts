import { z } from "zod";

// =============================================================================
// SSE Chat Stream Events
// =============================================================================

const ChatArtifactSchema = z.object({
  path: z.string(),
  media_type: z.string(),
  size: z.number().optional(),
});

export type ChatArtifact = z.infer<typeof ChatArtifactSchema>;

export const ChatStreamEventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("thinking"), content: z.string().optional() }),
  z.object({ type: z.literal("text"), content: z.string().optional() }),
  z.object({ type: z.literal("tool"), content: z.string().optional(), data: z.unknown().optional() }),
  z.object({ type: z.literal("tool_use"), content: z.string().optional() }),
  z.object({ type: z.literal("tool_result"), content: z.string().optional() }),
  z.object({ type: z.literal("status"), content: z.string().optional() }),
  z.object({ type: z.literal("session"), session_id: z.string() }),
  z.object({
    type: z.literal("done"),
    content: z.string().optional(),
    text: z.string().optional(),
    media: ChatArtifactSchema.array().optional(),
    media_files: ChatArtifactSchema.array().optional(),
    session_id: z.string().optional(),
    __llm_trace__: z.array(z.record(z.string(), z.unknown())).optional(),
  }),
  z.object({ type: z.literal("error"), message: z.string().optional() }),
]);

export type ChatStreamEvent = z.infer<typeof ChatStreamEventSchema>;

/**
 * Normalize legacy SSE event shapes before validation.
 * Maps "chunk"/"content" types to "text" so the schema accepts older server formats.
 */
export function parseChatStreamEvent(raw: Record<string, unknown>): ChatStreamEvent {
  // Normalize legacy type names
  if (raw.type === "chunk" || raw.type === "content") {
    raw = { ...raw, type: "text" };
  }
  return ChatStreamEventSchema.parse(raw);
}

// =============================================================================
// WebSocket Messages
// =============================================================================

export const WsMessageSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("status"),
    status: z.string().optional(),
    text: z.string().optional(),
    outputs: z.record(z.string(), z.unknown()).optional(),
    duration_ms: z.number().optional(),
  }),
  z.object({
    type: z.literal("chunk"),
    text: z.string().optional(),
    outputs: z.record(z.string(), z.unknown()).optional(),
    duration_ms: z.number().optional(),
  }),
  z.object({
    type: z.literal("done"),
    text: z.string().optional(),
    outputs: z.record(z.string(), z.unknown()).optional(),
    duration_ms: z.number().optional(),
  }),
  z.object({
    type: z.literal("error"),
    message: z.string().optional(),
    text: z.string().optional(),
  }),
]);

export type WsMessage = z.infer<typeof WsMessageSchema>;

// =============================================================================
// API Entities
// =============================================================================

export const ModelInfoSchema = z.object({
  id: z.string(),
  provider: z.string(),
  name: z.string(),
  category: z.string(),
  model: z.string(),
  api_url: z.string().nullable(),
  options: z.record(z.string(), z.unknown()),
  is_default: z.number(),
  created_at: z.string().optional(),
  updated_at: z.string().optional(),
});

export type ModelInfo = z.infer<typeof ModelInfoSchema>;

export const AgentInfoSchema = z.object({
  id: z.string(),
  name: z.string(),
  type: z.string(),
  description: z.string().nullable().optional(),
  role: z.string().nullable().optional(),
  model: z.string().nullable().optional(),
  is_default: z.number(),
  created_at: z.string(),
  updated_at: z.string(),
});

export type AgentInfo = z.infer<typeof AgentInfoSchema>;

export const LabSchema = z.object({
  id: z.string(),
  title: z.string(),
  description: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
});

export type Lab = z.infer<typeof LabSchema>;

export const LabDetailSchema = z.object({
  lab: LabSchema,
  sessions: z.array(z.record(z.string(), z.unknown())),
  session_id: z.string().nullable(),
});

export type LabDetail = z.infer<typeof LabDetailSchema>;

export const MCPServerSchema = z.object({
  id: z.string(),
  name: z.string(),
  transport: z.string(),
  url: z.string().optional(),
  auth: z.string().optional(),
  headers: z.string().optional(),
  command: z.string().optional(),
  args: z.string().optional(),
  envs: z.string().optional(),
  created_at: z.string(),
  updated_at: z.string(),
});

export type MCPServer = z.infer<typeof MCPServerSchema>;

export const ModelProviderSchema = z.object({
  id: z.string(),
  name: z.string(),
  type: z.string(),
  api_key: z.string().optional(),
  api_url: z.string().optional(),
  rate_limit: z
    .object({
      requests: z.number(),
      period: z.string(),
      margin: z.number(),
    })
    .optional(),
  pricing: z
    .object({
      currency: z.string().optional(),
      input: z.number().optional(),
      output: z.number().optional(),
      cached_input: z.number().optional(),
    })
    .optional(),
  created_at: z.string(),
  updated_at: z.string(),
});

export type ModelProvider = z.infer<typeof ModelProviderSchema>;

// =============================================================================
// Session / Options
// =============================================================================

export const SessionOptionsSchema = z.object({
  model: z.string().optional().nullable(),
  image_model: z.string().optional().nullable(),
  audio_model: z.string().optional().nullable(),
  video_model: z.string().optional().nullable(),
  system_prompt: z.string().optional().nullable(),
  skills: z.array(z.string()).optional(),
  mcp_servers: z.array(z.string()).optional(),
  agent: z.string().optional().nullable(),
  filter_tools: z.array(z.string()).optional(),
});

export type SessionOptions = z.infer<typeof SessionOptionsSchema>;

// =============================================================================
// Trace / Diagnostics
// =============================================================================

export const TraceIterationSchema = z.object({
  model: z.string().optional(),
  provider: z.string().optional(),
  request: z.unknown().optional(),
  response: z.unknown().optional(),
  message: z.unknown().optional(),
});

export type TraceIteration = z.infer<typeof TraceIterationSchema>;

// =============================================================================
// Form Schemas
// =============================================================================

export const ModelFormSchema = z.object({
  name: z.string().min(1, "Model name is required"),
  provider: z.string().min(1, "Provider is required"),
  category: z.enum(["Text", "Image", "Audio", "Video"]),
  model: z.string().min(1, "API model name is required"),
  api_url: z.string().nullable().optional(),
  options: z.record(z.string(), z.unknown()).optional(),
});

export type ModelFormData = z.infer<typeof ModelFormSchema>;

export const TaskFormSchema = z.object({
  name: z.string().min(1, "Task name is required"),
  description: z.string().optional(),
  skill: z.string().optional(),
  agent: z.string().optional(),
  schedule: z.string().optional(),
  input_schema: z.record(z.string(), z.unknown()).optional(),
});

export type TaskFormData = z.infer<typeof TaskFormSchema>;

// =============================================================================
// Utility
// =============================================================================

/**
 * Safely parse JSON and validate against a Zod schema.
 * Returns parsed data or throws with a descriptive error.
 */
export function safeJsonParse<T>(schema: z.ZodSchema<T>, json: string): T {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch {
    throw new Error("Failed to parse JSON");
  }
  return schema.parse(raw);
}

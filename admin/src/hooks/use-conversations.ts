import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, apiPost, apiDelete } from "@/lib/api";

export interface Conversation {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  turn_count: number;
  options?: Record<string, unknown>;
}

export interface ConversationTurn {
  id: number;
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
}

export function useConversations() {
  return useQuery({
    queryKey: ["conversations"],
    queryFn: () => api<Conversation[]>("/conversations"),
    refetchInterval: 30_000,
  });
}

export function useConversation(id: string | null) {
  return useQuery({
    queryKey: ["conversation", id],
    queryFn: () =>
      api<{ session: Conversation; turns: ConversationTurn[] }>(
        `/conversations/${encodeURIComponent(id!)}`,
      ),
    enabled: !!id,
    placeholderData: (prev) => prev,
  });
}

export function useCreateConversation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (title?: string) =>
      apiPost<{ id: string; title: string | null }>("/conversations", title ? { title } : undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
    },
  });
}

export function useDeleteConversation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/conversations/${encodeURIComponent(id)}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
    },
  });
}

export function useUpdateConversationTitle() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) =>
      api(`/conversations/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
    },
  });
}

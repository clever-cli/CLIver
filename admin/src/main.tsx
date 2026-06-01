import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router";
import { QueryClient, QueryClientProvider, QueryCache } from "@tanstack/react-query";
import { I18nProvider } from "@/i18n";
import { App } from "@/App";
import { SidebarProvider } from "@/contexts/sidebar-context";
import { AuthError } from "@/lib/api";
import LoginPage from "@/pages/login";
import ChatPage from "@/pages/chat";
import DashboardPage from "@/pages/dashboard";
import TaskListPage from "@/pages/tasks/list";
import TaskCreatePage from "@/pages/tasks/create";
import TaskDetailPage from "@/pages/tasks/detail";
import KeysList from "@/pages/keys/list";
import AgentListPage from "@/pages/agents/list";
import AgentDetailPage from "@/pages/agents/detail";
import SessionsPage from "@/pages/sessions";
import SessionDetailPage from "@/pages/session-detail";
import SkillsPage from "@/pages/skills";
import SkillDetailPage from "@/pages/skills/detail";
import SkillCreatePage from "@/pages/skills/create";
import ConfigPage from "@/pages/config";
import AdaptersList from "@/pages/adapters/list";
import LabListPage from "@/pages/labs/list";
import LabDetailPage from "@/pages/labs/detail";
import LabChatPage from "@/pages/labs/chat";
import MCPServersPage from "@/pages/mcp-servers/list";
import ModelsPage from "@/pages/models/list";
import "./globals.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 30_000 } },
  queryCache: new QueryCache({
    onError: (error) => {
      if (error instanceof AuthError) {
        window.location.href = "/admin/login";
      }
    },
  }),
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <I18nProvider>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Routes>
            <Route path="/admin/login" element={<LoginPage />} />
            <Route path="/admin" element={<SidebarProvider><App /></SidebarProvider>}>
              <Route index element={<Navigate to="dashboard" replace />} />
              <Route path="chat/:conversationId?" element={<ChatPage />} />
              <Route path="dashboard" element={<DashboardPage />} />
              <Route path="tasks" element={<TaskListPage />} />
              <Route path="tasks/new" element={<TaskCreatePage />} />
              <Route path="tasks/:name" element={<TaskDetailPage />} />
              <Route path="keys" element={<KeysList />} />
              <Route path="agents" element={<AgentListPage />} />
              <Route path="agents/:id" element={<AgentDetailPage />} />
              <Route path="sessions" element={<SessionsPage />} />
              <Route path="sessions/:id" element={<SessionDetailPage />} />
              <Route path="skills" element={<SkillsPage />} />
              <Route path="skills/new" element={<SkillCreatePage />} />
              <Route path="skills/:name" element={<SkillDetailPage />} />
              <Route path="labs" element={<LabListPage />} />
              <Route path="labs/:labId" element={<LabDetailPage />} />
              <Route path="labs/:labId/chat" element={<LabChatPage />} />
              <Route path="labs/:labId/chat/:sessionId" element={<LabChatPage />} />
              <Route path="models" element={<ModelsPage />} />
              <Route path="mcp-servers" element={<MCPServersPage />} />
              <Route path="adapters" element={<AdaptersList />} />
              <Route path="settings" element={<ConfigPage />} />
              <Route path="*" element={<Navigate to="/admin/dashboard" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </QueryClientProvider>
    </I18nProvider>
  </StrictMode>,
);

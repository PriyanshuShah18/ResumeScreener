const DEFAULT_API_BASE_URL = "http://localhost:8000";

export function getApiBaseUrl() {
  const customBase = import.meta.env.VITE_API_BASE_URL;
  if (!customBase || !customBase.trim()) {
    return DEFAULT_API_BASE_URL;
  }
  return customBase.trim().replace(/\/$/, "");
}

export function getErrorMessage(status, payload) {
  if (payload && typeof payload.detail === "string" && payload.detail.trim()) {
    return payload.detail;
  }
  if (payload && typeof payload.message === "string" && payload.message.trim()) {
    return payload.message;
  }
  return `Request failed with status ${status}.`;
}

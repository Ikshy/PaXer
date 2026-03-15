/**
 * UploadButton.tsx
 * Drag-and-drop file upload button that ingests a synthetic image and
 * optionally triggers inference immediately.
 */

import React, { useRef, useState } from "react";
import { Upload, Loader2 } from "lucide-react";
import { ingestImage, runInference } from "@/api/client";

interface UploadButtonProps {
  onComplete: () => void;
}

const UploadButton: React.FC<UploadButtonProps> = ({ onComplete }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setBusy(true);
    setStatusMsg("Ingesting…");
    try {
      const ingested = await ingestImage(file, "analyst_ui");
      setStatusMsg("Running inference…");
      await runInference(ingested.image_id, "analyst_ui");
      setStatusMsg("Done — item added to queue.");
      onComplete();
      setTimeout(() => setStatusMsg(null), 3000);
    } catch (err) {
      setStatusMsg(
        `Error: ${err instanceof Error ? err.message : "unknown error"}`
      );
    } finally {
      setBusy(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  };

  return (
    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
      <button
        onClick={() => inputRef.current?.click()}
        disabled={busy}
        aria-label="Upload synthetic image"
        style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          padding: "6px 14px",
          background: "#6366f1",
          color: "#fff",
          border: "none",
          borderRadius: "6px",
          fontSize: "12px",
          fontWeight: 600,
          cursor: busy ? "not-allowed" : "pointer",
          opacity: busy ? 0.6 : 1,
        }}
      >
        {busy ? <Loader2 size={13} style={{ animation: "spin 1s linear infinite" }} /> : <Upload size={13} />}
        Upload & Infer
      </button>

      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg"
        onChange={handleChange}
        style={{ display: "none" }}
        aria-hidden
      />

      {statusMsg && (
        <span
          aria-live="polite"
          style={{ color: "#94a3b8", fontSize: "12px" }}
        >
          {statusMsg}
        </span>
      )}
    </div>
  );
};

export default UploadButton;

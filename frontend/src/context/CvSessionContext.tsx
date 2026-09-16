import { createContext, useContext, useState, type ReactNode } from "react";
import type { CvUploadResponse } from "../api/types";

interface CvSessionValue {
  cvId: string | null;
  upload: CvUploadResponse | null;
  setUpload: (upload: CvUploadResponse) => void;
  clear: () => void;
}

const CvSessionContext = createContext<CvSessionValue | undefined>(undefined);

export function CvSessionProvider({ children }: { children: ReactNode }) {
  const [upload, setUploadState] = useState<CvUploadResponse | null>(null);

  const setUpload = (next: CvUploadResponse) => setUploadState(next);
  const clear = () => setUploadState(null);

  return (
    <CvSessionContext.Provider
      value={{ cvId: upload?.cv_id ?? null, upload, setUpload, clear }}
    >
      {children}
    </CvSessionContext.Provider>
  );
}

export function useCvSession(): CvSessionValue {
  const ctx = useContext(CvSessionContext);
  if (!ctx) throw new Error("useCvSession must be used within a CvSessionProvider");
  return ctx;
}

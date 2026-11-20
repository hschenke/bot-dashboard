"""Clientseitiges Components-v2-JavaScript für WebCrypto Device-PoP & Browser-Storage."""

from __future__ import annotations

AUTH_BRIDGE_HTML = '<div class="bot-dash-auth-bridge" aria-hidden="true"></div>'
AUTH_BRIDGE_CSS = ".bot-dash-auth-bridge { display: none !important; height: 0 !important; }"

AUTH_BRIDGE_JS = """
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  const TOKEN_RE = /^adm2:\\d+:[0-9a-f]{24}:[0-9a-f]{16}:[0-9a-f]{32}:[0-9a-f]{64}$/;
  const LEGACY_V1_RE = /^adm:\\d+:[0-9a-f]{64}$/;
  const IDB_NAME = "bot_dashboard_auth_v2", IDB_STORE = "device_keys", IDB_KEY_ID = "primary_p256";
  let cancelled = false;

  try {
    const c = parentElement?.closest?.('[data-testid="stElementContainer"]');
    if (c) { c.style.display = "none"; c.style.margin = "0"; c.style.padding = "0"; c.style.height = "0"; }
  } catch (_) {}

  const isValidToken = (v) => typeof v === "string" && TOKEN_RE.test(v.trim());
  const isLegacyV1 = (v) => typeof v === "string" && LEGACY_V1_RE.test(v.trim());
  const toB64Url = (buf) => {
    const bytes = new Uint8Array(buf);
    let bin = "";
    for (let i = 0; i < bytes.length; i += 1) bin += String.fromCharCode(bytes[i]);
    return btoa(bin).replace(/\\+/g, "-").replace(/\\//g, "_").replace(/=+$/g, "");
  };
  const getWins = () => {
    const wins = [];
    for (const cand of [window, window.parent, window.top]) {
      try { if (cand && cand.document && !wins.includes(cand)) wins.push(cand); } catch (_) {}
    }
    return wins;
  };
  const readCookie = (doc, name) => {
    try {
      const prefix = `${name}=`;
      for (const part of String(doc.cookie || "").split(";")) {
        const t = part.trim();
        if (t.startsWith(prefix)) {
          const raw = t.slice(prefix.length).trim();
          try { return decodeURIComponent(raw).trim(); } catch (_) { return raw; }
        }
      }
    } catch (_) {}
    return "";
  };
  const writeCookie = (doc, win, name, val, maxAge) => {
    try {
      const sec = win?.location?.protocol === "https:" ? "; Secure" : "";
      if (maxAge > 0 && val) {
        const exp = new Date(Date.now() + maxAge * 1000).toUTCString();
        doc.cookie = `${name}=${encodeURIComponent(val)}; max-age=${maxAge}; expires=${exp}; path=/; SameSite=Lax${sec}`;
      } else {
        doc.cookie = `${name}=; max-age=0; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=Lax${sec}`;
        doc.cookie = `${name}=; max-age=0; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
      }
    } catch (_) {}
  };
  const saveToken = (name, tok, maxAge) => {
    for (const w of getWins()) {
      try { w.localStorage?.setItem(name, tok); } catch (_) {}
      writeCookie(w.document, w, name, tok, maxAge);
    }
  };
  const clearToken = (name) => {
    for (const w of getWins()) {
      try { w.localStorage?.removeItem(name); } catch (_) {}
      writeCookie(w.document, w, name, "", 0);
    }
  };
  const purgeLegacy = (name) => {
    for (const w of getWins()) {
      try {
        const lsVal = String(w.localStorage?.getItem(name) || "").trim();
        if (lsVal && isLegacyV1(lsVal)) w.localStorage.removeItem(name);
      } catch (_) {}
      const ckVal = readCookie(w.document, name);
      if (ckVal && isLegacyV1(ckVal)) writeCookie(w.document, w, name, "", 0);
    }
  };
  const readToken = (name, candidate) => {
    const wins = getWins();
    for (const w of wins) {
      try { const s = String(w.localStorage?.getItem(name) || "").trim(); if (isValidToken(s)) return s; } catch (_) {}
    }
    for (const w of wins) {
      const ck = readCookie(w.document, name);
      if (isValidToken(ck)) return ck;
    }
    return isValidToken(candidate) ? candidate.trim() : "";
  };
  const openKeyDb = () => new Promise((resolve, reject) => {
    if (!window.indexedDB) { reject(new Error("no_idb")); return; }
    const req = window.indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(IDB_STORE)) req.result.createObjectStore(IDB_STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error("idb_open_failed"));
  });
  const idbReq = (req) => new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error || new Error("idb_failed"));
  });
  const getOrCreateKeyPair = async (createIfMissing) => {
    if (!window.isSecureContext || !window.crypto?.subtle || !window.indexedDB) return null;
    let db = null;
    try {
      db = await openKeyDb();
      const existing = await idbReq(db.transaction(IDB_STORE, "readonly").objectStore(IDB_STORE).get(IDB_KEY_ID));
      if (existing?.privateKey && existing?.publicKey) return existing;
      if (!createIfMissing) return null;
      const kp = await window.crypto.subtle.generateKey(
        { name: "ECDSA", namedCurve: "P-256" }, false, ["sign", "verify"]
      );
      const rec = { privateKey: kp.privateKey, publicKey: kp.publicKey };
      const tx = db.transaction(IDB_STORE, "readwrite");
      tx.objectStore(IDB_STORE).put(rec, IDB_KEY_ID);
      await new Promise((res, rej) => { tx.oncomplete = res; tx.onerror = rej; });
      return rec;
    } catch (_) { return null; } finally { try { db?.close(); } catch (_) {} }
  };
  const signText = async (priv, text) => {
    const sig = await window.crypto.subtle.sign(
      { name: "ECDSA", hash: "SHA-256" }, priv, new TextEncoder().encode(text)
    );
    return toB64Url(sig);
  };

  const action = String(data?.action || "noop");
  const cookieName = String(data?.cookieName || "bot_dashboard_auth_token");
  const maxAge = Number(data?.maxAge || 0);
  const token = String(data?.token || "").trim();
  const nonce = String(data?.nonce || "");
  const challenge = String(data?.challenge || "").trim();
  const candidate = String(data?.candidateToken || "").trim();

  if (action === "save" && isValidToken(token)) {
    saveToken(cookieName, token, maxAge);
  } else if (action === "clear") {
    clearToken(cookieName);
    window.__botDashRestoreNonce = "";
    window.__botDashBindNonce = "";
  } else if (action === "bind" && challenge) {
    if (nonce && window.__botDashBindNonce === nonce) return () => { cancelled = true; };
    window.__botDashBindNonce = nonce || challenge;
    (async () => {
      try {
        const pair = await getOrCreateKeyPair(true);
        if (cancelled) return;
        if (!pair) {
          setTriggerValue("restored", JSON.stringify({ kind: "bind", unsupported: true, challenge }));
          return;
        }
        const spki = await window.crypto.subtle.exportKey("spki", pair.publicKey);
        const sig = await signText(pair.privateKey, `bind:${challenge}`);
        if (!cancelled) {
          setTriggerValue("restored", JSON.stringify({
            kind: "bind", pubkey: toB64Url(spki), signature: sig, challenge,
          }));
        }
      } catch (_) {
        if (!cancelled) setTriggerValue("restored", JSON.stringify({ kind: "bind", unsupported: true, challenge }));
      }
    })();
  } else if (action === "restore") {
    if (nonce && window.__botDashRestoreNonce === nonce) return () => { cancelled = true; };
    purgeLegacy(cookieName);
    const saved = readToken(cookieName, candidate);
    if (saved) {
      window.__botDashRestoreNonce = nonce || "restored";
      const sid = saved.split(":")[2] || "";
      (async () => {
        try {
          const pair = await getOrCreateKeyPair(false);
          if (cancelled) return;
          if (!pair) {
            setTriggerValue("restored", JSON.stringify({ kind: "restore", token: saved, missing_key: true, challenge }));
            return;
          }
          const spki = await window.crypto.subtle.exportKey("spki", pair.publicKey);
          const sig = await signText(pair.privateKey, `restore:${challenge}:${sid}`);
          if (!cancelled) {
            setTriggerValue("restored", JSON.stringify({
              kind: "restore", token: saved, pubkey: toB64Url(spki), signature: sig, challenge,
            }));
          }
        } catch (_) {
          if (!cancelled) setTriggerValue("restored", JSON.stringify({ kind: "restore", token: saved, missing_key: true, challenge }));
        }
      })();
    }
  }
  return () => { cancelled = true; };
}
"""

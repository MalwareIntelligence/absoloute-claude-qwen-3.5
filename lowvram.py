"""

lowvram_gui_gguf.py

--------------------

Low-VRAM-Inferenz DIREKT auf einer GGUF-Datei, ueber llama-cpp-python.



Warum dieser Weg statt transformers/accelerate:

Deine GGUF laeuft nachweislich fehlerfrei mit llama.cpp (dafuer wurde sie ja

auch quantisiert/veroeffentlicht). Der vorherige Versuch ueber `transformers`

+ `accelerate`-Disk-Offloading nutzte eine komplett andere, sehr neue

Python-Implementierung von Qwen3.5s Linear-Attention, die noch diverse

Cache-/Reshape-Bugs hat. Dieses Script umgeht das Problem komplett, indem es

dieselbe Engine (llama.cpp) nutzt, die bei dir schon nachweislich funktioniert.



Das "wenig VRAM, grosses Modell"-Prinzip funktioniert hier ueber

n_gpu_layers: nur so viele Transformer-Layer wie angegeben werden auf die

GPU geladen, der Rest bleibt im normalen RAM. 0 = alles CPU/RAM,

sehr hohe Zahl (z.B. 999) = so viele wie moeglich auf die GPU.



NEU in dieser Version:

- Dateien anhaengen: Text-/Code-Dateien werden eingelesen und als Kontext

  vor den Prompt gehaengt (kein Tool-Call noetig, das Modell "sieht" den

  Inhalt direkt).

- Tool-Zugriff: Das Modell kann (falls das GGUF/der Chat-Template

  Funktions-Calling unterstuetzt, z.B. Qwen/Hermes-Stil) aktiv Tools

  aufrufen: Datei lesen, Verzeichnis auflisten, Datei schreiben und

  optional Python-Code ausfuehren (letzteres ist standardmaessig

  deaktiviert, da es beliebigen Code lokal ausfuehrt).



Benoetigte Pakete:

    pip install llama-cpp-python



Falls du eine NVIDIA-GPU nutzen willst, brauchst du das CUDA-Build von

llama-cpp-python (die normale pip-Version ist meist nur CPU). Installier-

Hinweis dazu steht unten im Abschnitt "GPU-Unterstuetzung", oder google

"llama-cpp-python CUDA wheel windows".



Start:

    python lowvram_gui_gguf.py

"""



import os

import sys

import ast

import operator

import json

import re

import lzma

import tarfile

import time

import datetime

import platform

import shutil

import urllib.request

import urllib.parse

import threading

import traceback

import subprocess

import tkinter as tk

from tkinter import ttk, filedialog, scrolledtext, messagebox



MAX_ATTACHMENT_CHARS = 20000  # pro Datei, damit der Kontext nicht explodiert

TOOL_LOOP_MAX_ITERATIONS = 6

RUN_PYTHON_TIMEOUT_SEC = 20

CONTEXT_SAFETY_MARGIN_TOKENS = 256  # Puffer zwischen Fenster-Budget und n_ctx

DEFAULT_CHAT_SESSIONS_DIR = os.path.join(os.getcwd(), "chat_sessions")

# Effort-Stufen (analog zu Claudes "effort"-Parameter): steuern, wie viel
# Denkzeit sich das Modell nehmen darf, plus eine kurze System-Anweisung,
# die dem Modell sagt, wie gruendlich es sein soll. Wird bei jedem Turn neu
# als System-Kontext eingefuegt (nicht persistiert), analog zu _inject_tools_prompt.
EFFORT_PRESETS = {
    "Low": {
        "max_reasoning_minutes": 1,
        "instruction": (
            "Effort: low. Answer briefly and directly, without long "
            "deliberation. Only think briefly for genuinely simple things, "
            "otherwise answer directly."
        ),
    },
    "Medium": {
        "max_reasoning_minutes": 5,
        "instruction": (
            "Effort: medium. Think as thoroughly as needed for a correct, "
            "solid answer -- but no longer than that."
        ),
    },
    "High": {
        "max_reasoning_minutes": 20,
        "instruction": (
            "Effort: high. Take as much time to think as needed. Check "
            "edge cases and alternatives thoroughly before answering."
        ),
    },
}
DEFAULT_EFFORT = "Medium"



# OpenSERP (https://github.com/karust/openserp) ist ein selbst gehosteter

# SERP-Proxy: er rendert die Suchmaschinenseite (Google/Bing/DuckDuckGo/...)

# selbst im Hintergrund-Browser und liefert sauberes JSON zurueck. Dadurch

# laeuft die eigentliche CAPTCHA-/Bot-Pruefung (falls eine auftritt) gegen

# den OpenSERP-Server statt gegen dieses Skript, und man bekommt strukturierte

# Ergebnisse statt fragilem HTML-Regex-Scraping. Standard-Port aus dem

# OpenSERP-Quick-Start (docker run ... serve -p 7000).

DEFAULT_OPENSERP_URL = "http://127.0.0.1:7000"

_openserp_base_url = ""  # von der GUI gesetzt, siehe set_openserp_base_url()





def set_openserp_base_url(url):

    """Wird von der GUI vor jedem Generierungs-Turn aufgerufen, um _run_web_

    search mitzuteilen, ob/wo ein OpenSERP-Server laeuft. Leerer String =

    OpenSERP nicht nutzen, direkt auf den DuckDuckGo-HTML-Fallback gehen."""

    global _openserp_base_url

    _openserp_base_url = (url or "").strip().rstrip("/")





class ThinkTagSplitter:

    """Reasoning-Modelle (Qwen3, QwQ, DeepSeek-R1-Distills usw.) liefern ihre

    Denkschritte bei eigenen/generischen GGUF-Chat-Templates NICHT ueber ein

    separates 'reasoning_content'-Feld -- llama-cpp-python unterstuetzt das

    nur fuer ein paar eingebaute Formate. Stattdessen steckt das Reasoning

    als <think>...</think> mitten im normalen Text. Diese Klasse trennt das

    live, Chunk fuer Chunk, wieder in ('reasoning', 'antwort') auf -- auch

    wenn ein Tag ueber zwei Stream-Chunks hinweg auseinandergerissen wird.

    """



    OPEN_TAG = "<think>"

    CLOSE_TAG = "</think>"

    MAX_TAG_LEN = max(len(OPEN_TAG), len(CLOSE_TAG))



    def __init__(self):

        self.buffer = ""

        self.in_think = False



    def feed(self, text):

        """Gibt eine Liste von (is_reasoning: bool, text: str) Stuecken

        zurueck, die sofort weitergegeben werden koennen."""

        self.buffer += text

        out = []

        while True:

            tag = self.CLOSE_TAG if self.in_think else self.OPEN_TAG

            idx = self.buffer.find(tag)

            if idx == -1:

                # Kein vollstaendiges Tag im Puffer. Am Ende koennte ein

                # angeschnittenes Tag stehen (z.B. Chunk endet mit "</th")

                # -- die letzten paar Zeichen sicherheitshalber zurueckhalten.

                safe_len = max(0, len(self.buffer) - (self.MAX_TAG_LEN - 1))

                if safe_len > 0:

                    out.append((self.in_think, self.buffer[:safe_len]))

                    self.buffer = self.buffer[safe_len:]

                break

            if idx > 0:

                out.append((self.in_think, self.buffer[:idx]))

            self.buffer = self.buffer[idx + len(tag):]

            self.in_think = not self.in_think

        return out



    def flush(self):

        out = []

        if self.buffer:

            out.append((self.in_think, self.buffer))

            self.buffer = ""

        return out


# ---------------------------------------------------------------------------
# GuiStreamBatcher: buendelt schnelle Streaming-Callbacks (ein Aufruf pro
# generiertem Token/Chunk) aus dem Worker-Thread zu EINEM Tk-Update, statt
# fuer jedes einzelne Token ein eigenes self.after()-Event samt
# Widget-Insert+Scroll einzuplanen. Ohne das konkurriert bei hohen
# Tokens/Sekunde der Tk-Mainloop-Thread (Widget-Redraw, .see('end') auf
# einer ueber die ganze Sitzung wachsenden Text-Box) staendig mit dem
# Generierungs-Thread um die GIL, was den effektiv erreichbaren Durchsatz
# spuerbar druecken kann. Mehrere add()-Aufrufe, die eintreffen bevor der
# naechste Tk-Frame verarbeitet wurde, landen im selben Puffer und werden
# mit einem einzigen flush_fn()-Aufruf ausgeliefert (Reihenfolge bleibt
# erhalten).
class GuiStreamBatcher:

    def __init__(self, tk_root, flush_fn):
        self._root = tk_root
        self._flush_fn = flush_fn
        self._lock = threading.Lock()
        self._buffer = []
        self._scheduled = False

    def add(self, text):
        if not text:
            return
        with self._lock:
            self._buffer.append(text)
            if self._scheduled:
                return
            self._scheduled = True
        self._root.after(0, self._flush)

    def _flush(self):
        with self._lock:
            chunk = "".join(self._buffer)
            self._buffer = []
            self._scheduled = False
        if chunk:
            self._flush_fn(chunk)






# ---------------------------------------------------------------------------

# ChatStore: legt den kompletten Chatverlauf verlustfrei auf der Platte ab,

# statt ihn im RAM zu halten. Jede Nachricht ist eine eigene kleine

# JSON-Datei; im Speicher liegt nur ein leichtgewichtiger Index (Dateiname,

# Rolle, Zeichenlaenge). Dadurch:

#   - RAM-Verbrauch bleibt auch bei riesigen Chats winzig (nur der Index).

#   - Kein "mehrfaches Komprimieren" moeglich, weil im Normalbetrieb ueberhaupt

#     nicht komprimiert wird -- es werden nur neue kleine Dateien angehaengt.

#     Das Korruptions-/Leer-Problem, das beim wiederholten Komprimieren

#     derselben Blase entsteht, kann so gar nicht auftreten.

#   - Fuer eine echte verlustfreie Archivierung (tragbar, wie eine .7z-Datei)

#     gibt es einen expliziten Export/Import ueber tar+xz (LZMA, dasselbe

#     Verfahren, das auch 7-Zip nutzt) -- immer als vollstaendige Frisch-

#     Kompression des aktuellen Ordnerinhalts, nie inkrementell.

# ---------------------------------------------------------------------------



class ChatStore:

    # Blockgroesse fuer Chunk-Aligned Eviction in build_window() (siehe dort).
    # 6 Nachrichten ~ 3 User/Assistant-Turnpaare -- grob an ChunkKVs Idee
    # angelehnt, ganze semantische Einheiten statt einzelner Tokens/Zeilen
    # zu evictieren, hier auf Nachrichtenebene uebertragen.
    _WINDOW_CHUNK_SIZE = 6

    def __init__(self, session_dir):

        self.session_dir = session_dir

        os.makedirs(session_dir, exist_ok=True)

        self.index_path = os.path.join(session_dir, "index.json")

        self.index = []  # Liste von {"file":..., "role":..., "n_chars":...}

        # In-Memory-Caches: Nachrichteninhalt aendert sich nach append() nie
        # mehr (kein Edit/Delete-Pfad im Code), daher koennen gelesene
        # Inhalte und einmal berechnete Tokenanzahlen pro Index sicher
        # wiederverwendet werden, statt bei JEDEM Turn erneut von der Platte
        # zu lesen und erneut zu tokenisieren. Das war vorher der
        # Haupt-Bottleneck VOR Generierungsbeginn (build_window liest+
        # tokenisiert bei jedem Turn den kompletten Verlaufs-Tail neu,
        # Kosten wachsen mit der Chatlaenge -> O(n) pro Turn, O(n^2) ueber
        # eine Sitzung).
        self._msg_cache = {}
        self._token_cache = {}
        # Chunk-Aligned Eviction (ChunkKV-Idee, siehe build_window): statt
        # bei jedem Turn erneut den exakten Trimm-Punkt frisch zu berechnen
        # (der durch kleine Schwankungen im Token-Budget -- z.B. je nach
        # aktueller 'Max. neue Tokens'-Einstellung oder context.md-Groesse
        # -- leicht hin- und herwackeln kann und dadurch das Praefix
        # unnoetig oft verschiebt), merkt sich die Sitzung den zuletzt
        # verwendeten Fenster-Start und verschiebt ihn nur in ganzen
        # Bloecken (_WINDOW_CHUNK_SIZE Nachrichten), sobald er wirklich
        # nicht mehr passt. Das haelt das an das Modell gesendete Praefix
        # ueber viele Turns hinweg stabil (ergaenzt den Praefix-Cache-Fix
        # in App._prepend_to_last_message) und reduziert gleichzeitig, wie
        # oft ueberhaupt neu prozessiert werden muss.
        self._window_start_idx = None

        self._load_or_rebuild_index()



    # ---- Index-Verwaltung ----

    def _load_or_rebuild_index(self):

        if os.path.exists(self.index_path):

            try:

                with open(self.index_path, "r", encoding="utf-8") as f:

                    self.index = json.load(f)

                return

            except Exception:

                pass  # Index kaputt/fehlt -> aus vorhandenen Dateien neu bauen

        self._rebuild_index_from_files()



    def _rebuild_index_from_files(self):

        self.index = []

        files = sorted(

            f for f in os.listdir(self.session_dir)

            if f.startswith("msg_") and f.endswith(".json")

        )

        for fname in files:

            path = os.path.join(self.session_dir, fname)

            try:

                with open(path, "r", encoding="utf-8") as fh:

                    msg = json.load(fh)

                self.index.append({

                    "file": fname,

                    "role": msg.get("role", "user"),

                    "n_chars": len(msg.get("content") or ""),

                })

            except Exception:

                continue  # einzelne kaputte Datei ueberspringen, Rest bleibt erhalten

        self._save_index()



    def _save_index(self):

        tmp_path = self.index_path + ".tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:

            json.dump(self.index, f, ensure_ascii=False)

        os.replace(tmp_path, self.index_path)  # atomar, verhindert kaputten Index bei Absturz



    # ---- Nachrichten schreiben/lesen ----

    def append(self, message):

        """message: dict mit mind. 'role' und 'content' (ggf. 'tool_calls',

        'tool_call_id', 'name'). Wird als eigene Datei persistiert."""

        n = len(self.index) + 1

        fname = f"msg_{n:08d}.json"

        path = os.path.join(self.session_dir, fname)

        tmp_path = path + ".tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:

            json.dump(message, f, ensure_ascii=False)

        os.replace(tmp_path, path)  # atomar -> kein korrupter Rest bei Absturz



        self.index.append({

            "file": fname,

            "role": message.get("role", "user"),

            "n_chars": len(message.get("content") or ""),

        })

        self._msg_cache[n - 1] = message

        self._save_index()



    def read(self, i):

        cached = self._msg_cache.get(i)
        if cached is not None:
            return cached

        path = os.path.join(self.session_dir, self.index[i]["file"])

        with open(path, "r", encoding="utf-8") as f:
            msg = json.load(f)

        self._msg_cache[i] = msg
        return msg



    def __len__(self):

        return len(self.index)



    def user_turn_count(self):

        return sum(1 for e in self.index if e["role"] == "user")



    def all_messages(self):

        """Vollstaendigen Verlauf laden (nur fuer Export/Anzeige -- fuers

        Modell wird build_window() genutzt, das NICHT alles laedt)."""

        return [self.read(i) for i in range(len(self.index))]



    # ---- Kontextfenster fuers Modell bauen (Disk -> nur benoetigter Teil in RAM) ----

    def build_window(self, budget_tokens, tokenize_fn):

        """Liest vom Ende her rueckwaerts so viele Nachrichten von der Platte,

        wie ins Token-Budget passen. System-Nachricht (Index 0, falls

        vorhanden) wird bevorzugt immer mitgenommen, notfalls gekuerzt.

        Der volle Verlauf auf der Platte bleibt davon unberuehrt."""

        n = len(self.index)

        if n == 0:

            return []



        has_system = self.index[0]["role"] == "system"

        system_msg = self.read(0) if has_system else None



        def cached_tokens(i, msg):
            # Inhalt unter Index i aendert sich nie mehr nach append() ->
            # einmal berechnete Tokenanzahl ist fuer die gesamte Sitzung
            # gueltig und muss nicht bei jedem Turn neu tokenisiert werden.
            t = self._token_cache.get(i)
            if t is None:
                t = tokenize_fn(msg.get("content") or "")
                self._token_cache[i] = t
            return t



        used = 0

        if system_msg is not None:

            sys_tokens = cached_tokens(0, system_msg)

            if sys_tokens > budget_tokens:

                # System-Nachricht (z.B. riesige Datei-Anhaenge) kuerzen,

                # damit ueberhaupt noch Platz fuer den Chat bleibt.

                content = system_msg.get("content") or ""

                keep_chars = max(500, len(content) * (budget_tokens // 2) // max(sys_tokens, 1))

                system_msg = dict(system_msg)

                system_msg["content"] = content[:keep_chars] + "\n[...fuer Kontextfenster gekuerzt, voll auf Platte erhalten...]"

                sys_tokens = tokenize_fn(system_msg["content"])

            used += sys_tokens



        min_start_idx = 1 if has_system else 0

        # --- Chunk-Aligned Eviction: erst pruefen, ob der zuletzt genutzte
        # Fenster-Start noch passt (Praefix bleibt dann exakt wie beim
        # letzten Turn -> voller Cache-Hit fuer den kompletten Verlauf bis
        # auf die juengste Nachricht). Nur falls nicht, in ganzen Chunks
        # weiterschieben statt Nachricht fuer Nachricht.
        CHUNK = self._WINDOW_CHUNK_SIZE

        def tokens_from(start):
            total = used
            for i in range(start, n):
                total += cached_tokens(i, self.read(i))
            return total

        start_idx = self._window_start_idx
        if start_idx is None or start_idx < min_start_idx or start_idx >= n:
            start_idx = min_start_idx

        if tokens_from(start_idx) > budget_tokens:
            # Passt nicht mehr -> in ganzen Bloecken vorschieben, bis es
            # wieder passt (oder nur noch die juengste Nachricht uebrig ist).
            while start_idx < n - 1 and tokens_from(start_idx) > budget_tokens:
                start_idx = min(start_idx + CHUNK, n - 1)
        # (Kein "else": passt der alte start_idx noch, bleibt er unangetastet
        # -- absichtlich KEIN Zurueckschieben Richtung Anfang, auch wenn
        # wieder mehr Budget frei waere, denn genau dieses Hin-und-Her waere
        # es, was das Praefix unnoetig instabil macht.)

        self._window_start_idx = start_idx

        tail = [self.read(i) for i in range(start_idx, n)]



        window = ([system_msg] if system_msg is not None else []) + tail

        return window



    # ---- Verlustfreie Archivierung (wie 7z: tar + LZMA) ----

    def archive_to(self, archive_path):

        """Packt den gesamten Sitzungsordner als einzelne, frisch komprimierte

        .tar.xz-Datei. Immer eine Kompression aus dem Klartext-Zustand heraus,

        nie auf einer bereits komprimierten Datei aufbauend."""

        with tarfile.open(archive_path, "w:xz") as tar:

            tar.add(self.session_dir, arcname=os.path.basename(self.session_dir))



    @staticmethod

    def load_archive(archive_path, extract_to_dir):

        with tarfile.open(archive_path, "r:xz") as tar:

            tar.extractall(extract_to_dir)

        # Name des obersten Ordners im Archiv ermitteln

        top_level = tarfile.open(archive_path, "r:xz").getnames()[0].split("/")[0]

        return os.path.join(extract_to_dir, top_level)





# ---------------------------------------------------------------------------

# Tool-Definitionen (OpenAI-kompatibles Function-Calling-Format, das

# llama-cpp-python an Modelle mit passendem Chat-Template durchreicht)

# ---------------------------------------------------------------------------



def parse_manual_tool_calls(text):

    """Manche Fine-Tunes (u.a. dieses Qwythos-Modell) geben Tool-Calls nicht

    ueber die native llama.cpp/OpenAI tool_calls-Struktur zurueck, sondern als

    reinen Text im eigenen Format, z.B.:



        <tool_call>{"name": "read_file", "arguments": {"path": "x.txt"}}</tool_call>

    oder:

        <tool_call>

        <function=read_file>

        <parameter=path>

        x.txt

        </parameter>

        </function>

        </tool_call>



    Diese Funktion erkennt beide Varianten per Regex, da llama-cpp-python das

    nicht automatisch in msg['tool_calls'] extrahiert, wenn das Chat-Template

    des Modells kein natives Function-Calling-Format nutzt."""

    calls = []

    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):

        block = block.strip()

        if not block:

            continue



        # Variante 1: JSON direkt im Block

        try:

            obj = json.loads(block)

            if isinstance(obj, dict) and "name" in obj:

                calls.append({"name": obj["name"], "arguments": obj.get("arguments", {}) or {}})

                continue

        except json.JSONDecodeError:

            pass



        # Variante 2: <function=name><parameter=key>wert</parameter>...</function>

        for func_match in re.finditer(r"<function=([\w\.\-]+)>(.*?)</function>", block, re.DOTALL):

            fname = func_match.group(1)

            fbody = func_match.group(2)

            args = {}

            for p_match in re.finditer(r"<parameter=([\w\.\-]+)>(.*?)</parameter>", fbody, re.DOTALL):

                key = p_match.group(1)

                val = p_match.group(2).strip()

                args[key] = val

            calls.append({"name": fname, "arguments": args})



    # Variante 3: <tool_code>...</tool_code> -- manche Modelle (z.B. an

    # Gemini/Codey-Konventionen trainierte Fine-Tunes) halluzinieren dieses

    # Python-Aufruf-Format statt des erwarteten <tool_call>-Tags, z.B.:

    #     <tool_code>

    #     list_dir(path=".")

    #     </tool_code>

    # Ohne diese Erkennung wird so ein Block bisher NICHT als Tool-Aufruf

    # erkannt -- das Modell haelt sich fuer fertig, die Antwort endet

    # kommentarlos direkt nach dem schliessenden Tag. Wir parsen den Python-

    # Ausdruck sicher per ast (kein eval!) und wandeln ihn in dasselbe

    # {"name":..., "arguments":...}-Format um wie die anderen Varianten.

    for block in re.findall(r"<tool_code>(.*?)</tool_code>", text, re.DOTALL):

        block = block.strip()

        if not block:

            continue

        # Nur die erste Zeile mit einem eigentlichen Funktionsaufruf nehmen,

        # falls das Modell Kommentare/mehrere Zeilen in den Block schreibt.

        call_line = next((ln.strip() for ln in block.splitlines() if "(" in ln), block)

        try:

            node = ast.parse(call_line, mode="eval").body

            if not isinstance(node, ast.Call):

                continue

            fname = node.func.id if isinstance(node.func, ast.Name) else None

            if not fname:

                continue

            args = {}

            for kw in node.keywords:

                if kw.arg is not None:

                    args[kw.arg] = ast.literal_eval(kw.value)

            # Falls Positionsargumente statt Keyword-Argumenten genutzt

            # wurden, koennen sie nicht sicher einem Parameternamen

            # zugeordnet werden -- solche Aufrufe werden uebersprungen,

            # statt falsche Argumente zu raten.

            if node.args:

                continue

            calls.append({"name": fname, "arguments": args})

        except (SyntaxError, ValueError):

            continue



    return calls





def build_tool_definitions(allow_read, allow_list, allow_write, allow_python, allow_diff,

                            allow_browser=False, allow_http=False, allow_search=False,

                            allow_calculate=True, allow_datetime=True, allow_append=False,

                            allow_delete=False, allow_move=False, allow_search_files=True,

                            allow_sysinfo=True):

    tools = []

    if allow_read:

        tools.append({

            "type": "function",

            "function": {

                "name": "read_file",

                "description": (

                    "Liest den Textinhalt einer Datei von der Festplatte. Optional "

                    "kann per 'start_line'/'end_line' NUR ein Zeilenbereich gelesen "

                    "werden (1-basiert, wie in einem Code-Editor) -- die Datei muss "

                    "dafuer nicht komplett geladen werden. Nuetzlich bei grossen "

                    "Dateien: z.B. erst mit start_line=1, end_line=50 einen Ausschnitt "

                    "ansehen, dann bei Bedarf gezielt weiterlesen. Ohne diese beiden "

                    "Parameter wird wie bisher die ganze Datei zurueckgegeben "

                    "(bei erneutem Lesen unveraendert/als Diff, um Kontext zu sparen)."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "path": {"type": "string", "description": "Pfad zur Datei"},

                        "start_line": {

                            "type": "integer",

                            "description": "1-basierte Startzeile (optional, Default 1 wenn end_line gesetzt ist)",

                        },

                        "end_line": {

                            "type": "integer",

                            "description": "1-basierte Endzeile inklusive (optional). -1 = bis Dateiende.",

                        },

                    },

                    "required": ["path"],

                },

            },

        })

    if allow_list:

        tools.append({

            "type": "function",

            "function": {

                "name": "list_dir",

                "description": "Listet Dateien und Unterordner in einem Verzeichnis auf.",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "path": {"type": "string", "description": "Verzeichnispfad"},

                    },

                    "required": ["path"],

                },

            },

        })

    if allow_write:

        tools.append({

            "type": "function",

            "function": {

                "name": "write_file",

                "description": "Schreibt Text in eine Datei (ueberschreibt sie).",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "path": {"type": "string", "description": "Zielpfad"},

                        "content": {"type": "string", "description": "Zu schreibender Inhalt"},

                    },

                    "required": ["path", "content"],

                },

            },

        })

    if allow_python:

        tools.append({

            "type": "function",

            "function": {

                "name": "run_python",

                "description": (

                    "Fuehrt ein kurzes Python-Snippet in einem separaten Prozess aus "

                    "und gibt stdout/stderr zurueck. Zeitlimit: "

                    f"{RUN_PYTHON_TIMEOUT_SEC}s."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "code": {"type": "string", "description": "Auszufuehrender Python-Code"},

                    },

                    "required": ["code"],

                },

            },

        })

    if allow_diff:

        tools.append({

            "type": "function",

            "function": {

                "name": "diff_files",

                "description": (

                    "Vergleicht zwei Textdateien und gibt einen unified diff "

                    "zurueck (wie 'diff -u' unter Linux, aber plattformunabhaengig "

                    "ueber Pythons difflib -- funktioniert also auch unter Windows)."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "path_a": {"type": "string", "description": "Pfad zur ersten (alten) Datei"},

                        "path_b": {"type": "string", "description": "Pfad zur zweiten (neuen) Datei"},

                        "context_lines": {

                            "type": "integer",

                            "description": "Anzahl Kontextzeilen um jede Aenderung (Default 3)",

                        },

                    },

                    "required": ["path_a", "path_b"],

                },

            },

        })

    if allow_append:

        tools.append({

            "type": "function",

            "function": {

                "name": "append_file",

                "description": "Haengt Text ans Ende einer Datei an, ohne den bisherigen Inhalt zu ueberschreiben (legt die Datei bei Bedarf neu an).",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "path": {"type": "string", "description": "Zielpfad"},

                        "content": {"type": "string", "description": "Anzuhaengender Text"},

                    },

                    "required": ["path", "content"],

                },

            },

        })

    if allow_delete:

        tools.append({

            "type": "function",

            "function": {

                "name": "delete_file",

                "description": "Loescht eine einzelne Datei unwiderruflich.",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "path": {"type": "string", "description": "Pfad zur zu loeschenden Datei"},

                    },

                    "required": ["path"],

                },

            },

        })

    if allow_move:

        tools.append({

            "type": "function",

            "function": {

                "name": "move_file",

                "description": "Verschiebt oder benennt eine Datei/einen Ordner um.",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "src": {"type": "string", "description": "Aktueller Pfad"},

                        "dst": {"type": "string", "description": "Neuer Pfad"},

                    },

                    "required": ["src", "dst"],

                },

            },

        })

    if allow_search_files:

        tools.append({

            "type": "function",

            "function": {

                "name": "search_files",

                "description": (

                    "Durchsucht Textdateien unter einem Verzeichnis rekursiv nach "

                    "einem Text/Regex-Muster (wie 'grep -r') und gibt Fundstellen "

                    "mit Datei, Zeilennummer und Zeileninhalt zurueck."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "directory": {"type": "string", "description": "Startverzeichnis"},

                        "pattern": {"type": "string", "description": "Suchtext oder Regex"},

                        "file_glob": {"type": "string", "description": "Dateimuster, z.B. '*.py'. Default '*'"},

                        "max_results": {"type": "integer", "description": "Max. Anzahl Treffer, Default 50"},

                    },

                    "required": ["directory", "pattern"],

                },

            },

        })

    if allow_http:

        tools.append({

            "type": "function",

            "function": {

                "name": "http_get",

                "description": (

                    "Ruft eine URL per einfachem HTTP-GET ab und gibt den "

                    "Text-Inhalt zurueck (kein JavaScript, kein echter Browser -- "

                    "schnell und leichtgewichtig fuer statische Seiten, APIs oder "

                    "JSON-Endpunkte). Fuer Seiten, die JS zum Laden brauchen, "

                    "stattdessen die browser_*-Tools verwenden."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "url": {"type": "string", "description": "Ziel-URL"},

                    },

                    "required": ["url"],

                },

            },

        })

    if allow_search:

        tools.append({

            "type": "function",

            "function": {

                "name": "web_search",

                "description": (

                    "Fuehrt eine Websuche aus und gibt eine Liste von Treffern "

                    "(Titel, URL, kurzer Textausschnitt) zurueck. Nutzt bevorzugt "

                    "einen konfigurierten OpenSERP-Server (robuster gegen CAPTCHA-"

                    "/Bot-Sperren, da dort ein echter Browser rendert); ohne "

                    "konfiguriertes OpenSERP faellt es automatisch auf einfaches "

                    "DuckDuckGo-HTML-Scraping zurueck."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "query": {"type": "string", "description": "Suchanfrage"},

                        "max_results": {"type": "integer", "description": "Max. Anzahl Ergebnisse, Default 5"},

                    },

                    "required": ["query"],

                },

            },

        })

    if allow_calculate:

        tools.append({

            "type": "function",

            "function": {

                "name": "calculate",

                "description": (

                    "Wertet einen mathematischen Ausdruck exakt aus (+ - * / // % ** "

                    "und Klammern), sicherer und zuverlaessiger als Kopfrechnen -- "

                    "immer benutzen statt Zahlen im Kopf/Text zu ueberschlagen."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "expression": {"type": "string", "description": "z.B. '(12.5 + 3) * 2 ** 4'"},

                    },

                    "required": ["expression"],

                },

            },

        })

    if allow_datetime:

        tools.append({

            "type": "function",

            "function": {

                "name": "get_datetime",

                "description": "Gibt das aktuelle Datum und die aktuelle Uhrzeit (lokal und UTC) zurueck.",

                "parameters": {"type": "object", "properties": {}, "required": []},

            },

        })

    if allow_sysinfo:

        tools.append({

            "type": "function",

            "function": {

                "name": "system_info",

                "description": "Gibt grundlegende Systeminfos zurueck: Betriebssystem, Python-Version, aktuelles Arbeitsverzeichnis, freier Festplattenspeicher.",

                "parameters": {"type": "object", "properties": {}, "required": []},

            },

        })

    if allow_browser:

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_navigate",

                "description": (

                    "Oeffnet eine URL in einem echten, sichtbaren Chrome-Browser "

                    "(Selenium). Startet den Browser beim ersten Aufruf automatisch "

                    "und haelt ihn danach ueber mehrere Tool-Aufrufe hinweg offen."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "url": {"type": "string", "description": "Ziel-URL, z.B. https://example.com"},

                    },

                    "required": ["url"],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_get_links",

                "description": (

                    "Listet die klickbaren Elemente (Links und Buttons) der aktuell "

                    "sichtbaren Seite durchnummeriert auf, mit ihrem sichtbaren Text "

                    "und (bei Links) Ziel-URL. Damit kann die Seite ohne Raten von "

                    "CSS-Selectoren erkundet werden: einfach die Nummer aus der Liste "

                    "an 'browser_click_index' uebergeben, um dorthin zu navigieren."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "max_links": {"type": "integer", "description": "Max. Anzahl Eintraege, Default 40"},

                    },

                    "required": [],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_click_index",

                "description": (

                    "Klickt das Element mit der angegebenen Nummer aus der zuletzt "

                    "per 'browser_get_links' geholten Liste an."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "index": {"type": "integer", "description": "Nummer aus der letzten browser_get_links-Liste"},

                    },

                    "required": ["index"],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_back",

                "description": "Geht im Browser-Verlauf eine Seite zurueck.",

                "parameters": {"type": "object", "properties": {}, "required": []},

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_forward",

                "description": "Geht im Browser-Verlauf eine Seite vor.",

                "parameters": {"type": "object", "properties": {}, "required": []},

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_refresh",

                "description": "Laedt die aktuelle Seite neu.",

                "parameters": {"type": "object", "properties": {}, "required": []},

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_current_state",

                "description": (

                    "Gibt aktuelle URL, Seitentitel und die Anzahl offener Tabs "

                    "zurueck -- nuetzlich, um sich nach einer Navigation zu "

                    "orientieren, bevor man get_text oder get_links aufruft."

                ),

                "parameters": {"type": "object", "properties": {}, "required": []},

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_wait_for",

                "description": (

                    "Wartet bis zu 'timeout' Sekunden, bis ein Element auf der Seite "

                    "erscheint. Nuetzlich bei Seiten, die Inhalte per JavaScript "

                    "nachladen -- vor dem naechsten Tool-Call aufrufen, statt blind "

                    "sofort auf ein noch nicht vorhandenes Element zuzugreifen."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "selector": {"type": "string", "description": "CSS-Selector (oder XPath/ID, siehe 'by')"},

                        "by": {

                            "type": "string",

                            "enum": ["css", "xpath", "id", "name", "link_text"],

                            "description": "Art des Selectors, Default 'css'",

                        },

                        "timeout": {"type": "number", "description": "Timeout in Sekunden, Default 10"},

                    },

                    "required": ["selector"],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_get_html",

                "description": (

                    "Gibt den rohen HTML-Quelltext der aktuellen Seite zurueck "

                    "(gekuerzt). Fuer Faelle, in denen get_text/get_links nicht "

                    "reichen, z.B. um Attribute, Tabellenstruktur oder verstecktes "

                    "Markup zu inspizieren."

                ),

                "parameters": {"type": "object", "properties": {}, "required": []},

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_select_option",

                "description": "Waehlt eine Option in einem <select>-Dropdown aus (per sichtbarem Text, Wert oder Index).",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "selector": {"type": "string", "description": "CSS-Selector (oder XPath/ID, siehe 'by') des <select>-Elements"},

                        "by": {

                            "type": "string",

                            "enum": ["css", "xpath", "id", "name", "link_text"],

                            "description": "Art des Selectors, Default 'css'",

                        },

                        "text": {"type": "string", "description": "Sichtbarer Text der Option"},

                        "value": {"type": "string", "description": "value-Attribut der Option"},

                        "index": {"type": "integer", "description": "Nullbasierter Index der Option"},

                    },

                    "required": ["selector"],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_press_key",

                "description": (

                    "Sendet eine Sondertaste (ENTER, ESCAPE, TAB, BACKSPACE, "

                    "ARROW_DOWN, ARROW_UP, PAGE_DOWN, PAGE_UP) an ein Element oder, "

                    "falls kein Selector angegeben ist, an das aktuell fokussierte "

                    "Element der Seite."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "key": {

                            "type": "string",

                            "enum": ["ENTER", "ESCAPE", "TAB", "BACKSPACE", "ARROW_DOWN", "ARROW_UP", "PAGE_DOWN", "PAGE_UP"],

                        },

                        "selector": {"type": "string", "description": "Optional: CSS-Selector des Zielelements"},

                        "by": {

                            "type": "string",

                            "enum": ["css", "xpath", "id", "name", "link_text"],

                            "description": "Art des Selectors, Default 'css'",

                        },

                    },

                    "required": ["key"],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_get_attribute",

                "description": "Liest ein HTML-Attribut (z.B. 'href', 'value', 'class', 'src') eines Elements aus.",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "selector": {"type": "string", "description": "CSS-Selector (oder XPath/ID, siehe 'by')"},

                        "attribute": {"type": "string", "description": "Name des Attributs"},

                        "by": {

                            "type": "string",

                            "enum": ["css", "xpath", "id", "name", "link_text"],

                            "description": "Art des Selectors, Default 'css'",

                        },

                    },

                    "required": ["selector", "attribute"],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_handle_alert",

                "description": (

                    "Bestaetigt oder bricht einen JavaScript-Alert/Confirm/Prompt-"

                    "Dialog ab, der die Seite gerade blockiert. Optional kann Text "

                    "in einen Prompt-Dialog eingegeben werden."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "accept": {"type": "boolean", "description": "true = OK/Bestaetigen, false = Abbrechen. Default true."},

                        "text": {"type": "string", "description": "Optional: Text fuer einen window.prompt()-Dialog"},

                    },

                    "required": [],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_switch_frame",

                "description": (

                    "Wechselt in ein eingebettetes iframe (per Selector oder Index) "

                    "oder mit default=true zurueck ins Hauptdokument. Manche Seiten "

                    "(z.B. eingebettete Formulare/Videos) liegen in einem iframe, "

                    "dessen Inhalt erst nach dem Wechsel per get_text/click sichtbar ist."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "selector": {"type": "string", "description": "CSS-Selector (oder XPath/ID, siehe 'by') des iframes"},

                        "by": {

                            "type": "string",

                            "enum": ["css", "xpath", "id", "name", "link_text"],

                            "description": "Art des Selectors, Default 'css'",

                        },

                        "index": {"type": "integer", "description": "Alternativ: nullbasierter Index des iframes auf der Seite"},

                        "default": {"type": "boolean", "description": "true = zurueck zum Hauptdokument wechseln"},

                    },

                    "required": [],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_new_tab",

                "description": "Oeffnet einen neuen Browser-Tab und wechselt zu ihm, optional mit direkter Ziel-URL.",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "url": {"type": "string", "description": "Optional: sofort zu ladende URL"},

                    },

                    "required": [],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_list_tabs",

                "description": "Listet alle offenen Tabs mit Nummer, Titel und URL auf; markiert den aktiven Tab.",

                "parameters": {"type": "object", "properties": {}, "required": []},

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_switch_tab",

                "description": "Wechselt zum Tab mit der angegebenen Nummer (siehe browser_list_tabs).",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "index": {"type": "integer", "description": "Tab-Nummer aus browser_list_tabs"},

                    },

                    "required": ["index"],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_close_tab",

                "description": "Schliesst einen Tab (Default: den aktuell aktiven) und wechselt danach zum naechsten verbleibenden.",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "index": {"type": "integer", "description": "Optional: Nummer des zu schliessenden Tabs, Default aktiver Tab"},

                    },

                    "required": [],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_execute_js",

                "description": (

                    "Fuehrt beliebigen JavaScript-Code im Kontext der aktuellen Seite "

                    "aus und gibt dessen Rueckgabewert zurueck (Vorsicht: sehr "

                    "maechtig, kann die Seite beliebig veraendern -- nur fuer Faelle, "

                    "die kein anderes Browser-Tool abdeckt, z.B. Werte auslesen, die "

                    "nicht als Attribut vorliegen)."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "code": {"type": "string", "description": "JS-Code, z.B. 'return document.title;'"},

                    },

                    "required": ["code"],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_click",

                "description": "Klickt auf ein Element der aktuell geladenen Seite.",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "selector": {"type": "string", "description": "CSS-Selector (oder XPath/ID, siehe 'by')"},

                        "by": {

                            "type": "string",

                            "enum": ["css", "xpath", "id", "name", "link_text"],

                            "description": "Art des Selectors, Default 'css'",

                        },

                    },

                    "required": ["selector"],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_type",

                "description": "Tippt Text in ein Eingabefeld der aktuell geladenen Seite, optional mit Enter/Absenden.",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "selector": {"type": "string", "description": "CSS-Selector (oder XPath/ID, siehe 'by')"},

                        "text": {"type": "string", "description": "Einzugebender Text"},

                        "by": {

                            "type": "string",

                            "enum": ["css", "xpath", "id", "name", "link_text"],

                            "description": "Art des Selectors, Default 'css'",

                        },

                        "submit": {

                            "type": "boolean",

                            "description": "Wenn true, wird nach der Eingabe Enter gedrueckt (z.B. Suchformular absenden).",

                        },

                    },

                    "required": ["selector", "text"],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_get_text",

                "description": (

                    "Liest den sichtbaren Text der aktuellen Seite (oder eines "

                    "einzelnen Elements, falls 'selector' angegeben wird) aus."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "selector": {"type": "string", "description": "Optional: CSS-Selector eines einzelnen Elements"},

                        "by": {

                            "type": "string",

                            "enum": ["css", "xpath", "id", "name", "link_text"],

                            "description": "Art des Selectors, Default 'css'",

                        },

                    },

                    "required": [],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_screenshot",

                "description": "Speichert einen Screenshot der aktuellen Seite als PNG-Datei auf der Platte.",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "path": {"type": "string", "description": "Zielpfad fuer die PNG-Datei"},

                    },

                    "required": ["path"],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_scroll",

                "description": "Scrollt die aktuelle Seite vertikal um die angegebene Pixelzahl (negativ = nach oben).",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "pixels": {"type": "integer", "description": "Anzahl Pixel, Default 800"},

                    },

                    "required": [],

                },

            },

        })

        tools.append({

            "type": "function",

            "function": {

                "name": "browser_close",

                "description": "Schliesst den vom Modell gesteuerten Browser wieder.",

                "parameters": {"type": "object", "properties": {}, "required": []},

            },

        })

    if tools:

        # Meta-Tool: liefert bei Bedarf die ausfuehrliche Anleitung (volle

        # Beschreibung, Parameterdetails) fuer eines oder mehrere der oben

        # aktiven Tools nach. So kann der System-Prompt standardmaessig

        # kurz bleiben (Tokenersparnis) und die KI holt sich Details nur,

        # wenn sie sie wirklich braucht -- laeuft im normalen Tool-Ablauf

        # automatisch mit, ohne dass die Antwort fuer den Nutzer unterbrochen wird.

        tools.append({

            "type": "function",

            "function": {

                "name": "get_tool_help",

                "description": (

                    "Gibt die ausfuehrliche Anleitung (volle Beschreibung, "

                    "alle Parameter mit Typ/Bedeutung, Sonderfaelle) fuer "

                    "eines oder mehrere der oben kurz gelisteten Tools "

                    "zurueck. Aufrufen, wenn die Kurzbeschreibung eines "

                    "Tools nicht reicht, um sicher zu sein, wie es benutzt "

                    "werden soll -- z.B. vor dem ersten Einsatz eines "

                    "riskanten Tools (write_file, delete_file, move_file, "

                    "run_python) oder bei unklaren Parametern."

                ),

                "parameters": {

                    "type": "object",

                    "properties": {

                        "tool_names": {

                            "type": "array",

                            "items": {"type": "string"},

                            "description": "Namen der Tools, zu denen die Anleitung gewuenscht ist, z.B. ['write_file', 'run_python'].",

                        },

                    },

                    "required": ["tool_names"],

                },

            },

        })

    return tools





def _short_tool_desc(desc, max_chars=110):

    """Kuerzt eine Tool-Beschreibung auf den ersten Satz (bzw. max_chars)

    fuer den kompakten Standard-Prompt. Die volle Beschreibung bleibt ueber

    get_tool_help abrufbar -- hier soll nur reichen, damit das Modell

    grob weiss, wofuer das Tool da ist."""

    text = " ".join(desc.split())

    first_sentence = text.split(". ")[0].rstrip(".")

    if len(first_sentence) > max_chars:

        first_sentence = first_sentence[:max_chars].rsplit(" ", 1)[0]

        return first_sentence + "…"

    return first_sentence + "."





def build_tools_system_prompt(tools):

    """Baut einen KOMPAKTEN System-Prompt: pro Tool nur Name + kurzer

    Einzeiler, keine vollen Parameterbeschreibungen. Details holt sich das

    Modell bei Bedarf selbst ueber das get_tool_help-Tool (siehe execute_tool

    / _get_tool_help_text) -- das laeuft im normalen Tool-Ablauf automatisch

    mit, ohne die Antwort an den Nutzer zu unterbrechen, spart aber im

    Normalfall (wenn kein Detailwissen noetig ist) Tokens gegenueber einem

    von vornherein vollstaendig ausformulierten Prompt.



    Die Namen/Kurzbeschreibungen werden direkt aus den Tool-Definitionen

    gezogen, damit sie nie von den tatsaechlich aktiven Tools abweichen.

    """

    if not tools:

        return ""



    lines = [

        "Du hast in dieser Sitzung Zugriff auf Tools (Funktionsaufrufe), mit "

        "denen du aktiv auf dem Rechner des Nutzers handeln kannst, statt "

        "nur aus dem Gedaechtnis zu antworten oder zu raten.",

        "",

        "Aktuell verfuegbare Tools (Kurzform):",

    ]



    active_names = []

    for t in tools:

        fn = t.get("function", {})

        name = fn.get("name", "?")

        active_names.append(name)

        short = _short_tool_desc(fn.get("description", ""))

        lines.append(f"- {name}: {short}")



    active_set = set(active_names)



    rules = [

        "Rufe ein Tool nur auf, wenn es fuer den naechsten Schritt "

        "tatsaechlich gebraucht wird -- nicht auf Verdacht, nicht mehrfach "

        "hintereinander mit identischen Argumenten.",

        "Die Kurzbeschreibungen oben nennen keine Parameter. Bevor du ein "

        "Tool zum ERSTEN Mal in dieser Sitzung benutzt oder wenn du dir bei "

        "Parametern/Verhalten unsicher bist, rufe get_tool_help(tool_names="

        "[\"<name>\", ...]) auf -- das liefert die vollstaendige Anleitung "

        "als normales Tool-Ergebnis nach, ohne deine Antwort an den Nutzer "

        "zu unterbrechen; danach machst du im selben Zug normal weiter.",

    ]



    write_like = active_set & {"write_file", "delete_file", "move_file", "append_file"}

    if write_like:

        rules.append(

            f"{', '.join(sorted(write_like))} veraendern echte Dateien auf "

            "der Festplatte des Nutzers"

            + (" -- insbesondere delete_file ist unwiderruflich" if "delete_file" in write_like else "")

            + ". Bei riskanten Aktionen kurz erklaeren, was du tust; im "

            "Zweifel vorher get_tool_help dazu holen."

        )



    if "run_python" in active_set:

        rules.append(

            "run_python fuehrt Code lokal aus (Zeitlimit beachten) -- nur "

            "fuer Faelle nutzen, die kein spezialisiertes Tool bereits "

            "abdeckt."

        )



    if active_set & {"browser_navigate", "browser_get_links", "browser_click_index",

                      "browser_back", "browser_forward", "browser_refresh",

                      "browser_current_state", "browser_wait_for", "browser_get_html",

                      "browser_select_option", "browser_press_key", "browser_get_attribute",

                      "browser_handle_alert", "browser_execute_js", "browser_click",

                      "browser_type", "browser_get_text", "browser_close"}:

        rules.append(

            "Zeigt eine Seite eine CAPTCHA- oder Bot-Schutzpruefung (das "

            "Tool meldet das ggf. automatisch als Hinweis zurueck), versuche "

            "NICHT, sie zu umgehen -- teile das dem Nutzer stattdessen "

            "ehrlich mit."

        )



    rules.append(

        "Nach Tool-Aufrufen immer normal in natuerlicher Sprache mit dem "

        "eigentlichen Ergebnis antworten -- die Tool-Aufrufe selbst sieht "

        "der Nutzer nicht direkt."

    )



    lines.append("")

    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))

    lines.append("Regeln fuer den Einsatz:\n" + numbered)

    return "\n".join(lines)





# ---------------------------------------------------------------------------

# FileStateCache: haelt pro gelesener Datei nur (mtime, Inhalt) im RAM.

# Wird dieselbe Datei unveraendert erneut per Tool gelesen -> Stub statt

# vollem Content (spart Tokens/Kontext). Aendert sich die Datei extern

# -> Diff wird injiziert statt des kompletten Texts. LRU-begrenzt, damit

# der RAM-Verbrauch auch bei vielen Dateien konstant bleibt.

# ---------------------------------------------------------------------------

import difflib

from collections import OrderedDict



class FileStateCache:

    def __init__(self, max_entries=100, max_bytes=25 * 1024 * 1024):

        self.max_entries = max_entries

        self.max_bytes = max_bytes

        self._store = OrderedDict()  # path -> (mtime, content)

        self._total_bytes = 0



    def _evict_if_needed(self):

        while self._store and (

            len(self._store) > self.max_entries or self._total_bytes > self.max_bytes

        ):

            _, (_, old_content) = self._store.popitem(last=False)

            self._total_bytes -= len(old_content)



    def read(self, path):

        """Liest path; gibt Text fuers Modell zurueck (Stub/Diff/Volltext)."""

        try:

            mtime = os.path.getmtime(path)

        except OSError as e:

            return f"[Konnte Datei nicht lesen: {e}]"



        with open(path, "r", encoding="utf-8", errors="replace") as f:

            content = f.read()

        truncated = content

        if len(truncated) > MAX_ATTACHMENT_CHARS:

            truncated = truncated[:MAX_ATTACHMENT_CHARS] + "\n[...gekuerzt...]"



        cached = self._store.get(path)

        if cached is not None:

            old_mtime, old_content = cached

            if old_mtime == mtime:

                self._store.move_to_end(path)

                return f"[unveraendert seit letztem Lesen: {path}]"

            diff = "\n".join(difflib.unified_diff(

                old_content.splitlines(), content.splitlines(),

                lineterm="", n=2,

            ))

            if len(diff) > MAX_ATTACHMENT_CHARS:

                diff = diff[:MAX_ATTACHMENT_CHARS] + "\n[...gekuerzt...]"

            self.set(path, mtime, content)

            return f"[Datei geaendert, Diff]:\n{diff}"



        self.set(path, mtime, content)

        return truncated



    def set(self, path, mtime, content):

        """Write-through: nach eigenem write_file aufrufen, damit der

        naechste Turn den eigenen Schreibzugriff nicht als externe

        Aenderung missversteht."""

        if path in self._store:

            _, old_content = self._store.pop(path)

            self._total_bytes -= len(old_content)

        self._store[path] = (mtime, content)

        self._total_bytes += len(content)

        self._evict_if_needed()





_file_state_cache = FileStateCache()





# ---------------------------------------------------------------------------

# SeleniumBrowserController: haelt einen echten Chrome-Browser am Leben, den

# das Modell per Tool-Calls steuern darf (Seite oeffnen, klicken, Text

# eintippen, Seiteninhalt/Text lesen, Screenshot speichern, schliessen).

# Der Browser wird erst beim ersten Tool-Aufruf lazy gestartet und bleibt

# danach ueber mehrere Tool-Calls/Turns hinweg offen, bis er explizit

# geschlossen wird oder das Programm endet.

#

# Benoetigt zusaetzlich:

#     pip install selenium

# sowie einen installierten Chrome/Chromium-Browser. Moderne Selenium-

# Versionen (>=4.6) laden den passenden chromedriver automatisch ueber

# Selenium Manager, ein manuelles Treiber-Setup ist normalerweise nicht

# noetig.

# ---------------------------------------------------------------------------



# Typische Textbausteine, die auf eine CAPTCHA- oder Bot-Schutzpruefung

# hindeuten (Cloudflare/Akamai/PerimeterX/DataDome/reCAPTCHA/hCaptcha usw.).

# Reine ERKENNUNG -- es wird nichts automatisiert geloest oder umgangen,

# das Modell soll die Situation nur ehrlich an den Nutzer melden.

_BOT_CHALLENGE_MARKERS = [

    "captcha", "recaptcha", "hcaptcha", "verify you are human",

    "verifying you are human", "checking your browser",

    "just a moment", "press and hold", "attention required",

    "unusual traffic", "automated access", "bot detection",

    "are you a robot", "security check", "ddos protection by",

    "access denied", "cf-chl", "cf_chl", "__cf_chl", "perimeterx",

    "datadome", "please stand by, while we are checking",

]





def _detect_bot_challenge(title, html_snippet):

    """Prueft Titel/HTML-Anfang oberflaechlich auf bekannte Bot-/CAPTCHA-

    Marker. Gibt None zurueck, wenn nichts gefunden wurde, sonst einen

    Hinweistext fuers Modell -- keine Umgehung, nur ehrliche Meldung."""

    haystack = f"{title or ''}\n{html_snippet or ''}".lower()

    hits = sorted({m for m in _BOT_CHALLENGE_MARKERS if m in haystack})

    if not hits:

        return None

    return (

        "[Hinweis: Diese Seite zeigt vermutlich eine CAPTCHA- oder "

        f"Bot-Schutzpruefung (erkannte Marker: {', '.join(hits[:3])}). "

        "Automatisiertes Umgehen solcher Pruefungen wird hier nicht "

        "unterstuetzt -- teile das dem Nutzer ehrlich mit, statt es zu "

        "versuchen zu umgehen. Der Browser ist sichtbar (kein Headless-"

        "Modus), der Nutzer kann die Pruefung ggf. selbst manuell im "

        "gleichen Fenster loesen; danach mit browser_current_state pruefen, "

        "ob es weitergeht.]"

    )





# ---------------------------------------------------------------------------



class SeleniumBrowserController:

    def __init__(self):

        self.driver = None

        self._last_links = []  # zuletzt per get_links gesammelte WebElemente, fuer click_index



    def _ensure_started(self, headless=False):

        if self.driver is not None:

            return

        try:

            from selenium import webdriver

            from selenium.webdriver.chrome.options import Options

        except ImportError as e:

            raise RuntimeError(

                "Selenium ist nicht installiert. Bitte 'pip install selenium' "

                "ausfuehren. Ausserdem wird ein installierter Chrome- oder "

                "Chromium-Browser benoetigt (der passende Treiber wird von "

                "Selenium >=4.6 automatisch geladen)."

            ) from e

        options = Options()

        if headless:

            options.add_argument("--headless=new")

        options.add_argument("--window-size=1280,900")

        self.driver = webdriver.Chrome(options=options)



    def _find(self, selector, by):

        from selenium.webdriver.common.by import By

        by_map = {

            "css": By.CSS_SELECTOR,

            "xpath": By.XPATH,

            "id": By.ID,

            "name": By.NAME,

            "link_text": By.LINK_TEXT,

        }

        return self.driver.find_element(by_map.get(by, By.CSS_SELECTOR), selector)



    def navigate(self, url, headless=False):

        self._ensure_started(headless=headless)

        if not url.startswith(("http://", "https://")):

            url = "https://" + url

        self.driver.get(url)

        result = f"Seite geladen: {self.driver.current_url}  |  Titel: {self.driver.title}"

        try:

            snippet = self.driver.page_source[:3000]

        except Exception:

            snippet = ""

        note = _detect_bot_challenge(self.driver.title, snippet)

        if note:

            result += "\n" + note

        return result



    def click(self, selector, by="css"):

        self._ensure_started()

        el = self._find(selector, by)

        el.click()

        return f"Element geklickt: {selector}"



    def type_text(self, selector, text, by="css", submit=False):

        self._ensure_started()

        el = self._find(selector, by)

        el.clear()

        el.send_keys(text)

        if submit:

            from selenium.webdriver.common.keys import Keys

            el.send_keys(Keys.RETURN)

        return f"Text eingegeben in: {selector}" + (" (mit Enter abgesendet)" if submit else "")



    def get_text(self, selector=None, by="css"):

        self._ensure_started()

        if selector:

            el = self._find(selector, by)

            text = el.text

        else:

            text = self.driver.find_element("tag name", "body").text

        if len(text) > MAX_ATTACHMENT_CHARS:

            text = text[:MAX_ATTACHMENT_CHARS] + "\n[...gekuerzt...]"

        return text or "(kein sichtbarer Text gefunden)"



    def screenshot(self, path):

        self._ensure_started()

        ok = self.driver.save_screenshot(path)

        return f"Screenshot gespeichert: {path}" if ok else "Screenshot fehlgeschlagen."



    def scroll(self, pixels=800):

        self._ensure_started()

        self.driver.execute_script(f"window.scrollBy(0, {int(pixels)});")

        return f"Um {pixels}px gescrollt."



    def get_links(self, max_links=40):

        self._ensure_started()

        elements = self.driver.find_elements("css selector", "a[href], button")

        self._last_links = []

        lines = []

        for el in elements:

            if len(self._last_links) >= max_links:

                break

            try:

                text = (el.text or el.get_attribute("aria-label") or "").strip()

                text = " ".join(text.split())

                if not text:

                    continue

                if not el.is_displayed():

                    continue

            except Exception:

                continue

            idx = len(self._last_links)

            self._last_links.append(el)

            href = ""

            try:

                href = el.get_attribute("href") or ""

            except Exception:

                pass

            tag = el.tag_name

            if href:

                lines.append(f"[{idx}] ({tag}) {text}  ->  {href}")

            else:

                lines.append(f"[{idx}] ({tag}) {text}")

        if not lines:

            return "Keine klickbaren Elemente mit sichtbarem Text gefunden."

        out = "\n".join(lines)

        if len(out) > MAX_ATTACHMENT_CHARS:

            out = out[:MAX_ATTACHMENT_CHARS] + "\n[...gekuerzt...]"

        return out



    def click_index(self, index):

        self._ensure_started()

        if not (0 <= index < len(self._last_links)):

            return (

                f"FEHLER: Index {index} ungueltig -- zuerst 'browser_get_links' "

                "aufrufen, um die aktuelle Liste zu holen."

            )

        el = self._last_links[index]

        try:

            el.click()

        except Exception:

            # Element evtl. ausserhalb des Viewports -> erst hinscrollen, dann klicken

            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)

            el.click()

        self._last_links = []  # Seite hat sich vermutlich geaendert, Cache verwerfen

        return f"Element [{index}] geklickt. Neue Seite ggf. mit browser_current_state pruefen."



    def back(self):

        self._ensure_started()

        self.driver.back()

        return f"Zurueck navigiert: {self.driver.current_url}"



    def forward(self):

        self._ensure_started()

        self.driver.forward()

        return f"Vorwaerts navigiert: {self.driver.current_url}"



    def refresh(self):

        self._ensure_started()

        self.driver.refresh()

        return f"Seite neu geladen: {self.driver.current_url}"



    def current_state(self):

        self._ensure_started()

        n_tabs = len(self.driver.window_handles)

        result = f"URL: {self.driver.current_url}\nTitel: {self.driver.title}\nOffene Tabs: {n_tabs}"

        try:

            snippet = self.driver.page_source[:3000]

        except Exception:

            snippet = ""

        note = _detect_bot_challenge(self.driver.title, snippet)

        if note:

            result += "\n" + note

        return result



    def wait_for(self, selector, by="css", timeout=10):

        self._ensure_started()

        from selenium.webdriver.support.ui import WebDriverWait

        from selenium.webdriver.support import expected_conditions as EC

        from selenium.common.exceptions import TimeoutException

        from selenium.webdriver.common.by import By

        by_map = {

            "css": By.CSS_SELECTOR, "xpath": By.XPATH, "id": By.ID,

            "name": By.NAME, "link_text": By.LINK_TEXT,

        }

        try:

            WebDriverWait(self.driver, float(timeout)).until(

                EC.presence_of_element_located((by_map.get(by, By.CSS_SELECTOR), selector))

            )

            return f"Element gefunden: {selector}"

        except TimeoutException:

            return f"Element NICHT gefunden innerhalb {timeout}s: {selector}"



    def get_html(self):

        self._ensure_started()

        html = self.driver.page_source

        note = _detect_bot_challenge(self.driver.title, html[:3000])

        if len(html) > MAX_ATTACHMENT_CHARS:

            html = html[:MAX_ATTACHMENT_CHARS] + "\n[...gekuerzt...]"

        if note:

            html += "\n" + note

        return html



    def select_option(self, selector, by="css", text=None, value=None, index=None):

        self._ensure_started()

        from selenium.webdriver.support.ui import Select

        el = self._find(selector, by)

        sel = Select(el)

        if value is not None:

            sel.select_by_value(value)

        elif text is not None:

            sel.select_by_visible_text(text)

        elif index is not None:

            sel.select_by_index(int(index))

        else:

            return "FEHLER: eines von 'text', 'value' oder 'index' angeben."

        return f"Option ausgewaehlt in: {selector}"



    def press_key(self, key, selector=None, by="css"):

        self._ensure_started()

        from selenium.webdriver.common.keys import Keys

        key_map = {

            "ENTER": Keys.ENTER, "ESCAPE": Keys.ESCAPE, "TAB": Keys.TAB,

            "BACKSPACE": Keys.BACKSPACE, "ARROW_DOWN": Keys.ARROW_DOWN,

            "ARROW_UP": Keys.ARROW_UP, "PAGE_DOWN": Keys.PAGE_DOWN, "PAGE_UP": Keys.PAGE_UP,

        }

        k = key_map.get(key.upper())

        if k is None:

            return f"FEHLER: unbekannte Taste '{key}'."

        el = self._find(selector, by) if selector else self.driver.switch_to.active_element

        el.send_keys(k)

        return f"Taste {key} gesendet."



    def get_attribute(self, selector, attribute, by="css"):

        self._ensure_started()

        el = self._find(selector, by)

        val = el.get_attribute(attribute)

        return f"{attribute} = {val!r}"



    def handle_alert(self, accept=True, text=None):

        self._ensure_started()

        try:

            alert = self.driver.switch_to.alert

        except Exception:

            return "Kein aktiver Alert/Dialog vorhanden."

        if text is not None:

            try:

                alert.send_keys(text)

            except Exception:

                pass

        if accept:

            alert.accept()

            return "Alert bestaetigt (OK)."

        alert.dismiss()

        return "Alert abgebrochen."



    def switch_frame(self, selector=None, by="css", index=None, default=False):

        self._ensure_started()

        if default:

            self.driver.switch_to.default_content()

            return "Zurueck zum Hauptdokument gewechselt."

        if index is not None:

            self.driver.switch_to.frame(int(index))

            return f"Zu iframe #{index} gewechselt."

        if selector:

            el = self._find(selector, by)

            self.driver.switch_to.frame(el)

            return f"Zu iframe '{selector}' gewechselt."

        return "FEHLER: 'selector', 'index' oder default=true angeben."



    def new_tab(self, url=None):

        self._ensure_started()

        self.driver.switch_to.new_window("tab")

        if url:

            if not url.startswith(("http://", "https://")):

                url = "https://" + url

            self.driver.get(url)

        return f"Neuer Tab geoeffnet: {self.driver.current_url}"



    def list_tabs(self):

        self._ensure_started()

        handles = self.driver.window_handles

        current = self.driver.current_window_handle

        lines = []

        for i, h in enumerate(handles):

            self.driver.switch_to.window(h)

            marker = " (aktiv)" if h == current else ""

            lines.append(f"[{i}] {self.driver.title}  -  {self.driver.current_url}{marker}")

        self.driver.switch_to.window(current)

        return "\n".join(lines)



    def switch_tab(self, index):

        self._ensure_started()

        handles = self.driver.window_handles

        if not (0 <= index < len(handles)):

            return f"FEHLER: Tab-Index {index} ungueltig (offene Tabs: {len(handles)})."

        self.driver.switch_to.window(handles[index])

        self._last_links = []

        return f"Zu Tab {index} gewechselt: {self.driver.current_url}"



    def close_tab(self, index=None):

        self._ensure_started()

        handles = self.driver.window_handles

        if index is None:

            target = self.driver.current_window_handle

        else:

            if not (0 <= index < len(handles)):

                return f"FEHLER: Tab-Index {index} ungueltig (offene Tabs: {len(handles)})."

            target = handles[index]

        self.driver.switch_to.window(target)

        self.driver.close()

        remaining = self.driver.window_handles

        if remaining:

            self.driver.switch_to.window(remaining[0])

        self._last_links = []

        return f"Tab geschlossen. Verbleibende Tabs: {len(remaining)}"



    def execute_js(self, code):

        self._ensure_started()

        result = self.driver.execute_script(code)

        return f"Ergebnis: {result!r}"



    def close(self):

        if self.driver is not None:

            try:

                self.driver.quit()

            finally:

                self.driver = None

        return "Browser geschlossen."





_browser_controller = SeleniumBrowserController()





# ---------------------------------------------------------------------------

# _safe_calculate: wertet einen mathematischen Ausdruck ueber den ast-Baum

# aus, statt eval() zu benutzen -- erlaubt sind nur Zahlen, Klammern und die

# Operatoren +, -, *, /, //, %, ** sowie unaeres Minus/Plus. Kein Funktions-

# aufruf, kein Namenszugriff moeglich, damit das Tool nicht zur Hintertuer

# fuer beliebigen Code wird.

# ---------------------------------------------------------------------------



_CALC_OPS = {

    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,

    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,

    ast.Mod: operator.mod, ast.Pow: operator.pow,

    ast.USub: operator.neg, ast.UAdd: operator.pos,

}





def _eval_calc_node(node):

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):

        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:

        return _CALC_OPS[type(node.op)](_eval_calc_node(node.left), _eval_calc_node(node.right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:

        return _CALC_OPS[type(node.op)](_eval_calc_node(node.operand))

    raise ValueError("Ausdruck enthaelt nicht erlaubte Elemente (nur Zahlen, + - * / // % ** und Klammern).")





def _safe_calculate(expression):

    try:

        tree = ast.parse(expression, mode="eval")

        result = _eval_calc_node(tree.body)

        return f"{expression} = {result}"

    except ZeroDivisionError:

        return "FEHLER: Division durch Null."

    except Exception as e:

        return f"FEHLER beim Auswerten von '{expression}': {e}"





def _run_web_search_openserp(query, max_results, base_url):

    """Fragt einen selbst gehosteten OpenSERP-Server ab (Megasearch-Endpunkt:

    fragt mehrere Suchmaschinen parallel ab und liefert sauberes JSON --

    keine eigene HTML-Regex-Fummelei, kein direktes CAPTCHA-Risiko fuer

    diesen Prozess, da der Browser/die Pruefung beim OpenSERP-Server

    laeuft). Wirft eine Exception bei Verbindungs-/Parserfehlern, damit der

    Aufrufer sauber auf den DuckDuckGo-Fallback umschalten kann."""

    params = {"text": query, "limit": max(1, min(max_results, 25))}

    url = f"{base_url}/mega/search?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, headers={"User-Agent": "LowVramGuiTool/1.0"})

    with urllib.request.urlopen(req, timeout=20) as resp:

        data = json.loads(resp.read().decode("utf-8", errors="replace"))



    results = []

    for item in data.get("results", []):

        title = (item.get("title") or "").strip()

        link = (item.get("url") or "").strip()

        snippet = (item.get("snippet") or "").strip()

        if not (title or link):

            continue

        results.append(f"{title}\n{link}\n{snippet}")

        if len(results) >= max_results:

            break



    if not results:

        return f"Keine Ergebnisse fuer '{query}' gefunden (OpenSERP)."

    return "\n\n".join(results)





def _run_web_search_ddg_html(query, max_results):

    """Fallback: HTML-Lite-Oberflaeche von DuckDuckGo per Regex-Scraping.

    Kein API-Key noetig, aber ohne echten Browser anfaelliger fuer

    Bot-/CAPTCHA-Pruefungen als der OpenSERP-Weg -- wird nur benutzt, wenn

    kein OpenSERP-Server konfiguriert ist oder dieser nicht erreichbar war."""

    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (LowVramGuiTool/1.0)"})

    with urllib.request.urlopen(req, timeout=15) as resp:

        html = resp.read().decode("utf-8", errors="replace")



    def strip_tags(s):

        return re.sub(r"<[^>]+>", "", s).strip()



    results = []

    for m in re.finditer(

        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'

        r'class="result__snippet"[^>]*>(.*?)</a>',

        html, re.DOTALL,

    ):

        raw_href, raw_title, raw_snippet = m.groups()

        href = urllib.parse.unquote(raw_href)

        # DuckDuckGo-Redirect-Link (//duckduckgo.com/l/?uddg=<echte-url>) aufloesen

        parsed = urllib.parse.urlparse(href)

        qs = urllib.parse.parse_qs(parsed.query)

        real_url = qs.get("uddg", [href])[0]

        title = strip_tags(raw_title)

        snippet = strip_tags(raw_snippet)

        results.append(f"{title}\n{real_url}\n{snippet}")

        if len(results) >= max_results:

            break



    if not results:

        return f"Keine Ergebnisse fuer '{query}' gefunden."

    return "\n\n".join(results)





def _run_web_search(query, max_results=5):

    """Sucht bevorzugt ueber einen konfigurierten OpenSERP-Server (siehe

    set_openserp_base_url) -- robuster gegen CAPTCHAs/Bot-Sperren, da der

    Server die Suchmaschinenseite selbst mit echtem Browser rendert.

    Ist kein OpenSERP konfiguriert oder schlaegt die Anfrage fehl, wird

    automatisch (mit kurzer Notiz) auf das alte DuckDuckGo-HTML-Scraping

    zurueckgefallen, damit web_search auch ohne laufenden OpenSERP-Server

    weiterhin funktioniert."""

    max_results = max(1, min(max_results, 15))



    if _openserp_base_url:

        try:

            return _run_web_search_openserp(query, max_results, _openserp_base_url)

        except Exception as e:

            fallback_note = (

                f"[Hinweis: OpenSERP unter {_openserp_base_url} nicht erreichbar "

                f"({e}) -- weiche auf DuckDuckGo-HTML-Fallback aus.]\n\n"

            )

            try:

                return fallback_note + _run_web_search_ddg_html(query, max_results)

            except Exception as e2:

                return f"FEHLER bei der Websuche (OpenSERP UND Fallback fehlgeschlagen): {e2}"



    try:

        return _run_web_search_ddg_html(query, max_results)

    except Exception as e:

        return f"FEHLER bei der Websuche: {e}"





def _read_file_line_range(path, start_line, end_line):

    """Liest nur die Zeilen [start_line, end_line] einer Datei (1-basiert,

    end_line inklusive; None/-1 = bis Dateiende). Streamt die Datei zeilen-

    weise und bricht ab, sobald end_line erreicht ist -- bei end_line < Datei-

    groesse wird also NICHT die ganze Datei gelesen, genau wie bei einem

    Datei-Viewer mit Zeilenbereich."""

    try:

        start = int(start_line) if start_line else 1

    except (TypeError, ValueError):

        start = 1

    if start < 1:

        start = 1



    end = None

    if end_line is not None:

        try:

            end_line = int(end_line)

        except (TypeError, ValueError):

            end_line = None

        if end_line is not None and end_line != -1:

            end = end_line



    lines_out = []

    last_line_num = 0

    reached_end = False

    try:

        with open(path, "r", encoding="utf-8", errors="replace") as f:

            for i, line in enumerate(f, start=1):

                if end is not None and i > end:

                    break

                last_line_num = i

                if i >= start:

                    lines_out.append(f"{i}\t{line.rstrip(chr(10))}")

            else:

                reached_end = True  # Schleife komplett durchlaufen -> Dateiende erreicht

    except OSError as e:

        return f"[Konnte Datei nicht lesen: {e}]"



    if not lines_out:

        return (

            f"[Keine Zeilen im Bereich {start}-{end if end is not None else '(Ende)'} "

            f"in {path} gefunden -- Datei hat evtl. weniger Zeilen als angefragt.]"

        )



    content = "\n".join(lines_out)

    if len(content) > MAX_ATTACHMENT_CHARS:

        content = content[:MAX_ATTACHMENT_CHARS] + "\n[...gekuerzt...]"



    footer = f"\n[{path}, Zeilen {start}-{last_line_num}"

    if reached_end:

        footer += f", Dateiende erreicht (insgesamt {last_line_num} Zeilen)]"

    else:

        footer += ", es gibt noch weitere Zeilen danach -- ggf. mit hoeherer start_line weiterlesen]"

    return content + footer





_ALL_TOOL_DOCS_CACHE = None





def _all_tool_docs():

    """Baut (einmalig, gecacht) eine Map name -> vollstaendiges function-

    Dict aus build_tool_definitions mit ALLEN Flags aktiv. Dient als

    Nachschlagewerk fuer get_tool_help, unabhaengig davon, welche Tools in

    der laufenden Sitzung gerade tatsaechlich angeschaltet sind."""

    global _ALL_TOOL_DOCS_CACHE

    if _ALL_TOOL_DOCS_CACHE is None:

        all_tools = build_tool_definitions(

            allow_read=True, allow_list=True, allow_write=True, allow_python=True,

            allow_diff=True, allow_browser=True, allow_http=True, allow_search=True,

            allow_calculate=True, allow_datetime=True, allow_append=True,

            allow_delete=True, allow_move=True, allow_search_files=True,

            allow_sysinfo=True,

        )

        _ALL_TOOL_DOCS_CACHE = {t["function"]["name"]: t["function"] for t in all_tools}

    return _ALL_TOOL_DOCS_CACHE





def _get_tool_help_text(tool_names):

    docs = _all_tool_docs()

    parts = []

    for n in tool_names:

        fn = docs.get(n)

        if not fn:

            parts.append(f"### {n}\n(Unbekannter Toolname -- kein Tool mit diesem Namen existiert.)")

            continue

        desc = " ".join(fn.get("description", "").split())

        props = fn.get("parameters", {}).get("properties", {})

        required = set(fn.get("parameters", {}).get("required", []))

        if props:

            arg_lines = []

            for pname, pinfo in props.items():

                mark = "" if pname in required else " (optional)"

                ptype = pinfo.get("type", "")

                pdesc = pinfo.get("description", "")

                arg_lines.append(f"  - {pname}{mark} [{ptype}]: {pdesc}")

            arg_block = "\n".join(arg_lines)

        else:

            arg_block = "  (keine Parameter)"

        parts.append(f"### {n}\n{desc}\nParameter:\n{arg_block}")

    return "\n\n".join(parts)





def execute_tool(name, arguments):

    """Fuehrt einen Tool-Call aus und gibt das Ergebnis als String zurueck."""

    try:

        if name == "get_tool_help":

            requested = arguments.get("tool_names")

            if isinstance(requested, str):

                requested = [requested]

            if not requested:

                return "FEHLER: 'tool_names' fehlt -- bitte mind. einen Toolnamen angeben, z.B. ['write_file']."

            return _get_tool_help_text(requested)



        if name == "read_file":

            path = arguments["path"]

            start_line = arguments.get("start_line")

            end_line = arguments.get("end_line")

            if start_line is not None or end_line is not None:

                return _read_file_line_range(path, start_line, end_line)

            return _file_state_cache.read(path)



        if name == "list_dir":

            path = arguments["path"]

            entries = os.listdir(path)

            return "\n".join(sorted(entries)) if entries else "(leer)"



        if name == "write_file":

            path = arguments["path"]

            content = arguments["content"]

            with open(path, "w", encoding="utf-8") as f:

                f.write(content)

            _file_state_cache.set(path, os.path.getmtime(path), content)

            return f"OK, {len(content)} Zeichen nach {path} geschrieben."



        if name == "diff_files":

            path_a = arguments["path_a"]

            path_b = arguments["path_b"]

            n = int(arguments.get("context_lines", 3))

            with open(path_a, "r", encoding="utf-8", errors="replace") as f:

                text_a = f.read()

            with open(path_b, "r", encoding="utf-8", errors="replace") as f:

                text_b = f.read()

            diff = "\n".join(difflib.unified_diff(

                text_a.splitlines(), text_b.splitlines(),

                fromfile=path_a, tofile=path_b, lineterm="", n=n,

            ))

            if not diff:

                return f"Keine Unterschiede zwischen {path_a} und {path_b}."

            if len(diff) > MAX_ATTACHMENT_CHARS:

                diff = diff[:MAX_ATTACHMENT_CHARS] + "\n[...gekuerzt...]"

            return diff



        if name == "append_file":

            path = arguments["path"]

            content = arguments["content"]

            with open(path, "a", encoding="utf-8") as f:

                f.write(content)

            try:

                with open(path, "r", encoding="utf-8", errors="replace") as f:

                    full = f.read()

                _file_state_cache.set(path, os.path.getmtime(path), full)

            except OSError:

                pass

            return f"OK, {len(content)} Zeichen an {path} angehaengt."



        if name == "delete_file":

            path = arguments["path"]

            os.remove(path)

            return f"OK, {path} geloescht."



        if name == "move_file":

            src = arguments["src"]

            dst = arguments["dst"]

            shutil.move(src, dst)

            return f"OK, {src} nach {dst} verschoben."



        if name == "search_files":

            directory = arguments["directory"]

            pattern = arguments["pattern"]

            file_glob = arguments.get("file_glob", "*")

            max_results = int(arguments.get("max_results", 50))

            try:

                rx = re.compile(pattern)

            except re.error:

                rx = re.compile(re.escape(pattern))

            hits = []

            for root, _dirs, files in os.walk(directory):

                for fname in files:

                    import fnmatch

                    if not fnmatch.fnmatch(fname, file_glob):

                        continue

                    fpath = os.path.join(root, fname)

                    try:

                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:

                            for lineno, line in enumerate(f, start=1):

                                if rx.search(line):

                                    hits.append(f"{fpath}:{lineno}: {line.strip()}")

                                    if len(hits) >= max_results:

                                        break

                    except OSError:

                        continue

                    if len(hits) >= max_results:

                        break

                if len(hits) >= max_results:

                    break

            if not hits:

                return f"Keine Treffer fuer '{pattern}' unter {directory}."

            out = "\n".join(hits)

            if len(out) > MAX_ATTACHMENT_CHARS:

                out = out[:MAX_ATTACHMENT_CHARS] + "\n[...gekuerzt...]"

            return out



        if name == "http_get":

            url = arguments["url"]

            if not url.startswith(("http://", "https://")):

                url = "https://" + url

            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (LowVramGuiTool/1.0)"})

            with urllib.request.urlopen(req, timeout=15) as resp:

                raw = resp.read()

                charset = resp.headers.get_content_charset() or "utf-8"

                text = raw.decode(charset, errors="replace")

            if len(text) > MAX_ATTACHMENT_CHARS:

                text = text[:MAX_ATTACHMENT_CHARS] + "\n[...gekuerzt...]"

            return text



        if name == "web_search":

            return _run_web_search(arguments["query"], int(arguments.get("max_results", 5)))



        if name == "calculate":

            return _safe_calculate(arguments["expression"])



        if name == "get_datetime":

            now_local = datetime.datetime.now()

            now_utc = datetime.datetime.now(datetime.timezone.utc)

            return (

                f"Lokal: {now_local.strftime('%Y-%m-%d %H:%M:%S')}\n"

                f"UTC:   {now_utc.strftime('%Y-%m-%d %H:%M:%S')}"

            )



        if name == "system_info":

            try:

                usage = shutil.disk_usage(os.getcwd())

                free_gb = usage.free / (1024 ** 3)

                disk_line = f"Freier Speicherplatz (aktuelles Laufwerk): {free_gb:.1f} GB"

            except OSError:

                disk_line = "Freier Speicherplatz: nicht ermittelbar"

            return (

                f"Betriebssystem: {platform.system()} {platform.release()}\n"

                f"Python: {platform.python_version()}\n"

                f"Arbeitsverzeichnis: {os.getcwd()}\n"

                f"{disk_line}"

            )



        if name == "run_python":

            code = arguments["code"]

            result = subprocess.run(

                [sys.executable, "-c", code],

                capture_output=True, text=True,

                timeout=RUN_PYTHON_TIMEOUT_SEC,

            )

            out = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\nreturncode: {result.returncode}"

            if len(out) > MAX_ATTACHMENT_CHARS:

                out = out[:MAX_ATTACHMENT_CHARS] + "\n[...gekuerzt...]"

            return out



        if name == "browser_navigate":

            return _browser_controller.navigate(arguments["url"])



        if name == "browser_get_links":

            return _browser_controller.get_links(int(arguments.get("max_links", 40)))



        if name == "browser_click_index":

            return _browser_controller.click_index(int(arguments["index"]))



        if name == "browser_back":

            return _browser_controller.back()



        if name == "browser_forward":

            return _browser_controller.forward()



        if name == "browser_refresh":

            return _browser_controller.refresh()



        if name == "browser_current_state":

            return _browser_controller.current_state()



        if name == "browser_wait_for":

            return _browser_controller.wait_for(

                arguments["selector"], arguments.get("by", "css"),

                float(arguments.get("timeout", 10)),

            )



        if name == "browser_get_html":

            return _browser_controller.get_html()



        if name == "browser_select_option":

            return _browser_controller.select_option(

                arguments["selector"], arguments.get("by", "css"),

                arguments.get("text"), arguments.get("value"), arguments.get("index"),

            )



        if name == "browser_press_key":

            return _browser_controller.press_key(

                arguments["key"], arguments.get("selector"), arguments.get("by", "css")

            )



        if name == "browser_get_attribute":

            return _browser_controller.get_attribute(

                arguments["selector"], arguments["attribute"], arguments.get("by", "css")

            )



        if name == "browser_handle_alert":

            return _browser_controller.handle_alert(

                bool(arguments.get("accept", True)), arguments.get("text")

            )



        if name == "browser_switch_frame":

            return _browser_controller.switch_frame(

                arguments.get("selector"), arguments.get("by", "css"),

                arguments.get("index"), bool(arguments.get("default", False)),

            )



        if name == "browser_new_tab":

            return _browser_controller.new_tab(arguments.get("url"))



        if name == "browser_list_tabs":

            return _browser_controller.list_tabs()



        if name == "browser_switch_tab":

            return _browser_controller.switch_tab(int(arguments["index"]))



        if name == "browser_close_tab":

            idx = arguments.get("index")

            return _browser_controller.close_tab(int(idx) if idx is not None else None)



        if name == "browser_execute_js":

            return _browser_controller.execute_js(arguments["code"])



        if name == "browser_click":

            return _browser_controller.click(

                arguments["selector"], arguments.get("by", "css")

            )



        if name == "browser_type":

            return _browser_controller.type_text(

                arguments["selector"], arguments["text"],

                arguments.get("by", "css"), bool(arguments.get("submit", False)),

            )



        if name == "browser_get_text":

            return _browser_controller.get_text(

                arguments.get("selector"), arguments.get("by", "css")

            )



        if name == "browser_screenshot":

            return _browser_controller.screenshot(arguments["path"])



        if name == "browser_scroll":

            return _browser_controller.scroll(int(arguments.get("pixels", 800)))



        if name == "browser_close":

            return _browser_controller.close()



        return f"Unbekanntes Tool: {name}"

    except subprocess.TimeoutExpired:

        return f"FEHLER: run_python hat das Zeitlimit ({RUN_PYTHON_TIMEOUT_SEC}s) ueberschritten."

    except Exception as e:

        return f"FEHLER bei Tool '{name}': {e}"





def read_attachment(path):

    """Liest eine angehaengte Datei als Text, mit Laengenbegrenzung."""

    try:

        with open(path, "r", encoding="utf-8", errors="replace") as f:

            content = f.read()

    except Exception as e:

        return f"[Konnte Datei nicht lesen: {e}]"

    if len(content) > MAX_ATTACHMENT_CHARS:

        content = content[:MAX_ATTACHMENT_CHARS] + "\n[...gekuerzt...]"

    return content





try:
    import numpy as _np
except ImportError:
    _np = None


class TokenRecycler:
    """Token Recycling (Luo et al., 2024, "Turning Trash into Treasure:
    Accelerating LLM Inference via Token Recycling"): baut waehrend der
    gesamten Sitzung eine einfache Adjazenz-Statistik Token -> haeufigster
    Folge-Token auf, aus ALLEN bisher gesehenen Tokens (Prompt + eigene
    Antworten). Kostet nur ein paar Python-Dicts im RAM -- kein zusaetzliches
    VRAM, kein zweites Modell. Dient als letzter Fallback, wenn weder der
    Fortsetzungs-Cache noch der Suffix-Index (siehe unten) einen Treffer
    liefern."""

    def __init__(self):
        self.table = {}  # token -> {next_token: count}

    def observe(self, seq):
        for a, b in zip(seq, seq[1:]):
            counter = self.table.get(a)
            if counter is None:
                counter = {}
                self.table[a] = counter
            counter[b] = counter.get(b, 0) + 1

    def propose(self, last_token, length):
        out = []
        cur = last_token
        for _ in range(length):
            counter = self.table.get(cur)
            if not counter:
                break
            cur = max(counter, key=counter.get)
            out.append(cur)
        return out


class SuffixHashIndex:
    """Vereinfachte, Hashmap-basierte Variante des Suffix-Automaten aus
    SAM-Decoding (Hu et al., 2024, "A Unified Framework for Retrieval-Based
    Speculative Decoding via a Suffix Automaton"): ein echter Suffix-Automat
    braucht O(1) Update/Query, ist dafuer aber deutlich komplexerer Code.
    Stattdessen wird hier je n-Gramm-Groesse ein Dict[n-Gramm -> letzte
    Position] gepflegt. Findet damit -- wie das Original -- den laengsten
    passenden Kontext-Suffix irgendwo in der GESAMTEN bisherigen Token-
    Historie, nicht nur (wie das urspruengliche Prompt-Lookup-Decoding) in
    einem festen kleinen Lookup-Fenster. Kostet etwas mehr RAM (ein paar
    Dicts mit int-Tupeln als Keys) -- kein VRAM, keine GPU.

    Da llama-cpp-python den Draft-Model-Aufruf mit den vollstaendigen,
    bereits tokenisierten input_ids macht (inklusive angehaengter Dateien/
    Tool-Ausgaben, siehe MAX_ATTACHMENT_CHARS oben, und inklusive der vom
    Modell selbst bereits erzeugten Tokens dieses Turns), deckt dieser
    Index automatisch zwei weitere Papers mit ab, ohne eigenen Code dafuer:
    Lookahead Decoding (Fu et al., 2024) -- Treffer koennen auch aus dem
    bisherigen Antworttext dieses Turns kommen -- und REST (He et al.,
    2024) -- Treffer koennen auch aus angehaengtem Dateiinhalt kommen, der
    faktisch als Retrieval-Datastore wirkt, ohne separaten Prozess."""

    def __init__(self, orders=(8, 6, 4, 3, 2)):
        self.orders = orders
        self.tables = {n: {} for n in orders}
        self.seq = []

    def extend(self, new_tokens):
        if not new_tokens:
            return
        start = len(self.seq)
        self.seq.extend(new_tokens)
        total = len(self.seq)
        for n in self.orders:
            table = self.tables[n]
            lo = max(start - n + 1, 0)
            hi = total - n + 1
            for i in range(lo, hi):
                key = tuple(self.seq[i:i + n])
                pos = i + n
                if pos >= total:
                    # Das ist der GERADE ERST entstandene Kontext-Suffix
                    # selbst -- dafuer gibt es noch keine Fortsetzung. Nicht
                    # ueber einen evtl. schon vorhandenen, echten (aelteren)
                    # Treffer fuer dasselbe n-Gramm druebermerkeln, sonst
                    # wird ein brauchbarer Match durch einen wertlosen
                    # Platzhalter ersetzt.
                    table.setdefault(key, pos)
                else:
                    table[key] = pos  # neuester ECHTER Treffer gewinnt

    def propose(self, length):
        for n in self.orders:
            if len(self.seq) < n:
                continue
            key = tuple(self.seq[-n:])
            pos = self.tables[n].get(key)
            if pos is not None and pos < len(self.seq):
                return self.seq[pos:pos + length]
        return []


class ContinuationCache:
    """Inspiriert von Ouroboros (Zhao et al., 2024, "Ouroboros: Speculative
    Decoding with Large Model Enhanced Drafting"): sobald ein Kandidaten-
    Vorschlag (aus welcher Quelle auch immer) ganz oder teilweise vom
    Hauptmodell akzeptiert wurde, wird die akzeptierte Fortsetzung unter
    dem Hash des davorstehenden Kontext-Suffixes abgelegt. Taucht genau
    dieser Suffix spaeter wieder auf -- auch in einem SPAETEREN Turn,
    typisch bei Tool-Ausgaben, Wiederholungen, aehnlichen Prompts --, kann
    direkt die bereits verifizierte, oft laengere Fortsetzung vorgeschlagen
    werden, statt erneut ueber den (kuerzeren) Suffix-Index zu gehen.
    Reines Python-Dict im RAM, kein zusaetzliches Modell/VRAM.

    capacity=200000 (vorher 20000): ein Eintrag ist ein int-Tupel (Key,
    ~6 Tokens) plus eine kurze Token-Liste (Fortsetzung, ~2-24 Tokens) --
    auch bei 200000 Eintraegen bleibt das im niedrigen zweistelligen MB-
    Bereich RAM, nichts im Vergleich zu einem echten KV-Cache. Die alte
    20000er-Grenze war zu niedrig gewaehlt und hat bei langen Sessions mit
    viel Wiederholung (z.B. lange Tool-/Code-Loops) fuer eine bereits
    ueberschriebene, nuetzliche Fortsetzung sorgen koennen, bevor der
    Kontext-Suffix sie erneut gebraucht haette. SuffixHashIndex oben ist
    ohnehin schon unbegrenzt (waechst mit der gesamten Token-Historie)."""

    def __init__(self, key_len=6, capacity=200000):
        self.key_len = key_len
        self.capacity = capacity
        self.table = {}
        self._order = []

    def _key(self, seq):
        if len(seq) < self.key_len:
            return None
        return tuple(seq[-self.key_len:])

    def remember(self, seq_before, accepted_continuation):
        if not accepted_continuation:
            return
        key = self._key(seq_before)
        if key is None:
            return
        if key not in self.table:
            if len(self._order) >= self.capacity:
                oldest = self._order.pop(0)
                self.table.pop(oldest, None)
            self._order.append(key)
        self.table[key] = accepted_continuation

    def propose(self, seq, length):
        key = self._key(seq)
        if key is None:
            return []
        cont = self.table.get(key)
        return cont[:length] if cont else []


class AdaptiveDraftLength:
    """Inspiriert von SpecDec++ (He et al., 2024, "SpecDec++: Boosting
    Speculative Decoding via Adaptive Candidate Lengths"): statt einer
    fixen num_pred_tokens-Zahl wird ein gleitender Durchschnitt der
    tatsaechlich akzeptierten Kandidaten-Laenge gefuehrt und die naechste
    angeforderte Laenge daran angepasst -- kurz bei wenig Wiederholung
    (spart nutzlose Verify-Rechenzeit), laenger bei viel woertlicher
    Wiederholung (z.B. Code-Diffs, Tool-Output-Zitate). Reine Heuristik,
    keine trainierten Parameter, kein zusaetzlicher Speicher."""

    def __init__(self, min_len=2, max_len=24, start=10, ema_alpha=0.3):
        self.min_len = min_len
        self.max_len = max_len
        self.current = start
        self.ema_alpha = ema_alpha
        self._ema_accept = float(start) / 2

    def update(self, accepted_len, requested_len):
        if requested_len <= 0:
            return
        self._ema_accept = (
            self.ema_alpha * accepted_len + (1 - self.ema_alpha) * self._ema_accept
        )
        target = int(round(self._ema_accept * 1.5)) + 1
        self.current = max(self.min_len, min(self.max_len, target))

    def length(self):
        return self.current


class MultiPaperDraftModel:
    """Ersatz fuer den fruaeheren MTP-Zweig. MTP (Multi-Token Prediction)
    braucht trainierte Zusatzkoepfe -- also eine eigene MTP-GGUF-Variante
    des Modells UND mehr VRAM fuer deren Gewichte. Fuer dieses Modell gibt
    es gar keine MTP-Variante, und selbst wenn: mehr VRAM war ausdruecklich
    nicht gewuenscht. Diese Klasse kombiniert stattdessen mehrere
    modellfreie, rein CPU/RAM-seitige Kandidaten-Generatoren aus der
    Speculative-/Lookup-Decoding-Literatur -- alle teilen sich dasselbe
    draft_model-Interface, das llama-cpp-python schon fuer
    LlamaPromptLookupDecoding nutzt (siehe llama_cpp.llama_speculative).
    Verifiziert wird weiterhin ganz normal vom bereits geladenen
    Hauptmodell selbst -- kein zusaetzliches VRAM, keine zweite GPU, kein
    zweites Modell.

    Kombinierte Papers (alle kombinierbar, alle ohne Zusatz-VRAM):
      1. Prompt Lookup Decoding (Saxena/apoorvumang, 2023) -- Grundidee,
         hier durch (2) verallgemeinert (nicht mehr nur festes Fenster).
      2. SAM-Decoding (Hu et al., 2024) -- SuffixHashIndex, Suche ueber
         die gesamte bisherige Token-Historie statt festem Lookup-Fenster.
      3. Lookahead Decoding (Fu et al., 2024) -- n-Gramm-Pool-Teil: der
         SuffixHashIndex wird laufend auch mit selbst generierten Tokens
         gefuettert, nicht nur dem urspruenglichen Prompt.
      4. REST (He et al., 2024) -- Retrieval-Datastore-Idee: angehaengte
         Dateien/Tool-Ausgaben landen automatisch im selben Suffix-Index,
         kein separater Prozess noetig (siehe SuffixHashIndex-Docstring).
      5. Ouroboros (Zhao et al., 2024) -- ContinuationCache: einmal
         verifizierte, laengere Fortsetzungen werden turnuebergreifend
         fuer denselben Kontext-Suffix wiederverwendet.
      6. Token Recycling (Luo et al., 2024) -- TokenRecycler als letzter
         Fallback (haeufigster Folge-Token), falls kein Suffix-Treffer da ist.
      7. SpecDec++ (He et al., 2024) -- AdaptiveDraftLength: passt die
         angeforderte Kandidatenlaenge laufend an die Trefferquote an.

      8. Speculative Checkpointing (srogmann, ggml-org/llama.cpp PR
         #19493, "Server: Speculative Checkpointing", gemerged 2026-04-19)
         -- ersetzt Kandidat Nr. 8 einer frueheren Fassung dieser Liste,
         der wegen fehlender VRAM-freier Kompatibilitaet mit Hybrid-/SSM-
         Architekturen (Qwen3.5 & Co.) ausgeschlossen war. Kernidee: statt
         einen verworfenen Draft per llama_memory_seq_rm (partielle
         KV-Entfernung) zurueckzunehmen -- was bei rekurrentem State
         (Mamba/SSM/Hybrid-Linear-Attention) strukturell nicht geht, siehe
         HYBRID_SSM_ARCH_MARKERS weiter unten --, wird vor dem Draft-
         Versuch ein vollstaendiger State-Checkpoint gespeichert und bei
         Ablehnung komplett restauriert. In llama.cpp selbst nur fuer
         llama-server/-cli in common/speculative.cpp umgesetzt, nicht fuer
         llama-cpp-python's generate()-Schleife -- deshalb hier nachgebaut
         als RecurrentStateCheckpoint (siehe dort) und als Ersatz fuer die
         fruehere komplette Abschaltung von Speculative Decoding bei
         Hybrid-/SSM-Modellen verwendet.

    Bewusst weiterhin NICHT aufgenommen -- zwei Kandidaten aus der
    Recherche, die entweder Zusatzgewichte+VRAM brauchen oder einen
    C-seitigen Hook voraussetzen, den llama-cpp-python (Stand dieser
    Version) auf der High-Level-Llama-Klasse nicht exponiert (dasselbe
    Problem wie beim urspruenglichen MTP-Hook):
      9. Medusa (Cai et al., 2024) -- trainierte Zusatzkoepfe, extra VRAM.
     10. EAGLE / EAGLE-3 (Li et al., 2024) -- trainierter Draft-Kopf,
         braucht eigene GGUF-Konvertierung + Zusatz-VRAM (siehe
         llama.cpp/docs/speculative.md, --spec-type draft-eagle3).
     11. Self-Speculative-/LayerSkip-Decoding (Zhang et al., 2024,
         "Draft & Verify"; Elhoushi et al., 2024, "LayerSkip") -- braucht
         zwar kein Zusatzmodell/VRAM, aber Kontrolle darueber, einzelne
         Transformer-Layer waehrend des Forward-Pass zu ueberspringen --
         dafuer gibt es aktuell keinen Python-Hook auf der Llama-Klasse.
    """

    def __init__(self, num_pred_tokens=10, min_pred_tokens=2, max_pred_tokens=24, log_fn=None):
        self.suffix_index = SuffixHashIndex()
        self.continuations = ContinuationCache()
        self.recycler = TokenRecycler()
        self.adaptive = AdaptiveDraftLength(
            min_len=min_pred_tokens, max_len=max_pred_tokens, start=num_pred_tokens
        )
        self.log = log_fn or (lambda msg: None)
        self._last_seq_len = 0
        self._last_proposal = []
        self._last_proposal_start = 0
        self.hits = {"continuation": 0, "suffix": 0, "recycle": 0, "miss": 0}

    def clear(self):
        """Bei 'Neuer Chat' aufrufen -- verhindert, dass Muster aus einer
        voellig anderen vorherigen Sitzung/Thema Fehltreffer erzeugen."""
        self.__init__(
            num_pred_tokens=self.adaptive.current,
            min_pred_tokens=self.adaptive.min_len,
            max_pred_tokens=self.adaptive.max_len,
            log_fn=self.log,
        )

    def _observe_progress(self, input_ids):
        seq = [int(t) for t in input_ids]
        new_len = len(seq)
        if new_len < self._last_seq_len:
            # Neuer Turn mit kuerzerem/anderem Prefix (z.B. nach Fenster-
            # Eviction) -- Zaehler zuruecksetzen, restliche Historie bleibt
            # in suffix_index/recycler/continuations trotzdem nutzbar.
            self._last_seq_len = 0
            self._last_proposal = []
        if new_len > self._last_seq_len:
            newly_added = seq[self._last_seq_len:new_len]
            if self._last_proposal:
                accepted = 0
                for i, tok in enumerate(newly_added):
                    if i < len(self._last_proposal) and tok == self._last_proposal[i]:
                        accepted += 1
                    else:
                        break
                self.adaptive.update(accepted, len(self._last_proposal))
                if accepted > 0:
                    prefix_before = seq[:self._last_proposal_start]
                    self.continuations.remember(prefix_before, newly_added[:accepted + 1])
            self.recycler.observe(seq[max(0, self._last_seq_len - 1):new_len])
            self.suffix_index.extend(newly_added)
        self._last_seq_len = new_len
        return seq

    def __call__(self, input_ids, **kwargs):
        seq = self._observe_progress(input_ids)
        length = self.adaptive.length()
        self._last_proposal_start = len(seq)

        cont = self.continuations.propose(seq, length)
        source = "continuation"
        if not cont:
            cont = self.suffix_index.propose(length)
            source = "suffix"
        if not cont and seq:
            cont = self.recycler.propose(seq[-1], length)
            source = "recycle"
        if not cont:
            source = "miss"

        self.hits[source] += 1
        self._last_proposal = cont
        if _np is not None:
            return _np.array(cont, dtype=_np.intc)
        return cont


class RecurrentStateCheckpoint:
    """Macht Speculative Decoding fuer Hybrid-/SSM-Modelle (Qwen3.5 & Co.,
    siehe HYBRID_SSM_ARCH_MARKERS unten) nutzbar, obwohl deren rekurrenter
    State -- anders als ein normaler Transformer-KV-Cache -- nicht per
    llama_memory_seq_rm partiell zurueckgenommen werden kann.

    Gleiches Prinzip wie das im April 2026 in llama.cpp gemergte
    "Speculative Checkpointing" (ggml-org/llama.cpp PR #19493,
    common/speculative.cpp): vor jedem Draft-Versuch wird der komplette
    State per llama_state_seq_get_data() gesichert; wird der Draft (ganz
    oder teilweise) abgelehnt, wird der State per llama_state_seq_set_data()
    komplett wiederhergestellt, statt (kaputt) partiell entfernt zu werden.
    llama.cpp selbst bietet das nur fuer llama-server/-cli -- llama-cpp-
    python's eigene generate()-Schleife ruft bei einem abgelehnten Token
    weiterhin bedingungslos self._ctx.kv_cache_seq_rm(-1, n, -1) auf (siehe
    llama_cpp/llama.py), was bei rekurrentem State fehlschlaegt bzw. zu
    einem vollen Re-Prefill fuehrt -- genau das Gegenteil dessen, was
    Speculative Decoding erreichen soll. Diese Klasse patcht daher gezielt
    genau diesen einen Aufruf auf der jeweiligen LlamaContext-Instanz.

    RAM-Kosten: die rekurrente State-Groesse ist FEST (haengt anders als
    ein klassischer KV-Cache NICHT von der Kontextlaenge ab) -- typ. wenige
    hundert KB bis niedrige einstellige MB, je nach Modell/Layeranzahl. Ein
    Snapshot kostet also konstanten, kleinen RAM, waechst nicht mit der
    Konversationslaenge und braucht kein zusaetzliches VRAM."""

    def __init__(self, llama_obj, log_fn=None):
        self.llama_obj = llama_obj
        self.log = log_fn or (lambda msg: None)
        self._ctx_p = llama_obj._ctx.ctx
        self._seq_id = 0
        self._snapshot = None
        self._snapshot_pos = None
        self.supported = self._probe()

    def _probe(self):
        try:
            import ctypes
            import llama_cpp as llama_cpp_mod
            self._ctypes = ctypes
            self._get_size = llama_cpp_mod.llama_state_seq_get_size
            self._get_data = llama_cpp_mod.llama_state_seq_get_data
            self._set_data = llama_cpp_mod.llama_state_seq_set_data
            return True
        except AttributeError as e:
            self.log(
                "Hinweis: llama_state_seq_get/set_data fehlt in dieser "
                f"llama-cpp-python-Version ({e}) -- Checkpoint-basiertes "
                "Speculative Decoding fuer Hybrid-/SSM-Modelle nicht "
                "verfuegbar."
            )
            return False

    def save(self, position):
        """Vor einem Draft-Versuch aufrufen. position = aktuelle Token-
        Position, damit try_restore() spaeter erkennt, ob der Rollback-
        Aufruf tatsaechlich zu diesem Snapshot gehoert."""
        if not self.supported:
            return
        try:
            size = self._get_size(self._ctx_p, self._seq_id)
            buf = (self._ctypes.c_uint8 * size)()
            self._get_data(self._ctx_p, buf, size, self._seq_id)
            self._snapshot = buf
            self._snapshot_pos = position
        except Exception as e:
            self.log(f"Checkpoint-Snapshot fehlgeschlagen: {e}")
            self._snapshot = None
            self._snapshot_pos = None

    def try_restore(self, target_position):
        """Wird vom gepatchten kv_cache_seq_rm aufgerufen (p0-Argument als
        target_position). True = passender Snapshot wiederhergestellt, der
        Original-Aufruf (kaputte partielle Entfernung) wird uebersprungen.
        False = kein passender Snapshot (z.B. Rollback aus anderem Grund,
        nicht von einem unserer Draft-Versuche) -- Aufrufer soll dann auf
        die urspruengliche Logik zurueckfallen."""
        if not self.supported or self._snapshot is None:
            return False
        if target_position != self._snapshot_pos:
            return False
        try:
            self._set_data(self._ctx_p, self._snapshot, len(self._snapshot), self._seq_id)
            return True
        except Exception as e:
            self.log(f"Checkpoint-Restore fehlgeschlagen: {e}")
            return False
        finally:
            self._snapshot = None
            self._snapshot_pos = None


class _CheckpointingDraftModel:
    """Duenner Wrapper um das eigentliche draft_model (MultiPaperDraftModel):
    sichert vor jedem Vorschlag einen RecurrentStateCheckpoint-Snapshot an
    der aktuellen Position. Reine Weiterleitung sonst -- keine Aenderung am
    Vorschlagsverhalten selbst.

    WICHTIG (Bugfix): llama-cpp-python's generate()-Schleife ruft den
    Draft-Model-Hook mit self.input_ids[: self.n_tokens + len(tokens)] auf
    (siehe llama_cpp/llama.py, Llama.generate) -- also einer Sequenz, die
    um len(tokens) LAENGER ist als die tatsaechliche KV-Cache-Position
    self.n_tokens, an der der State im Moment dieses Aufrufs wirklich
    steht (die neuen 'tokens' wurden ja noch nicht evaluiert). Der
    urspruengliche Code hier rief checkpoint.save(len(input_ids)) auf --
    das speicherte also eine um len(tokens) zu grosse Position zum
    Snapshot. Bei einem spaeteren Rollback (kv_cache_seq_rm mit
    p0 = self.n_tokens nach Ablehnung) passte target_position dadurch NIE
    zu self._snapshot_pos, try_restore() gab dauerhaft False zurueck, und
    der Patch fiel jedes Mal auf den urspruenglichen (bei rekurrentem
    State kaputten) kv_cache_seq_rm zurueck -- der Checkpoint griff
    dadurch faktisch nie, egal ob Speculative Decoding aktiv war oder
    (nach einem Retry) deaktiviert wurde. Fix: die tatsaechliche
    KV-Position direkt von der Llama-Instanz lesen (llama_obj.n_tokens)
    statt sie aus der Laenge der uebergebenen input_ids abzuleiten."""

    def __init__(self, inner, checkpoint, llama_obj):
        self.inner = inner
        self.checkpoint = checkpoint
        self.llama_obj = llama_obj

    def __call__(self, input_ids, **kwargs):
        self.checkpoint.save(self.llama_obj.n_tokens)
        return self.inner(input_ids, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


# Architektur-Namen (GGUF general.architecture / general.name), bei denen
# das Modell einen rekurrenten/SSM-artigen State (Mamba, Mamba2, RWKV,
# Hybrid-Linear-Attention wie Qwen3.5/Qwen3-Next, Jamba, ...) statt eines
# reinen Transformer-KV-Cache verwendet. Bei diesen Architekturen ist der
# interne State NICHT wie ein normaler KV-Cache partiell zuruecknehmbar --
# llama.cpp kann nach einer verworfenen Spekulation nicht einfach ein paar
# Positionen "zurueckspulen", weil es keinen expandierten Cache mehr gibt,
# sondern nur den bereits verrechneten, komprimierten Zustand. Speculative
# Decoding (Kandidaten vorschlagen -> ggf. verwerfen) setzt aber genau
# dieses Zuruecknehmen voraus. Ergebnis, falls man es trotzdem versucht:
# "the tokens ... have inconsistent sequence positions" / "X < Y" /
# "partial kv removal not supported, re-evaluating full prompt" -- meist
# nicht nur einmalig, sondern bei praktisch jedem Turn erneut, weil die
# gleiche Ursache (Draft-Modell + rekurrenter State) bei jedem neuen
# Prefill wieder zuschlaegt.
HYBRID_SSM_ARCH_MARKERS = (
    "mamba", "mamba2", "rwkv", "rwkv6", "rwkv7",
    "jamba", "zamba", "griffin", "recurrentgemma",
    "qwen3next", "qwen3-next", "qwen3_next",
    "qwen3.5", "qwen35",
    "plamo2", "lfm2", "nemotron-h", "nemotronh",
)

# Modelle mit M-RoPE (Multimodal RoPE: getrennte Positions-Achsen fuer
# Text/Zeit/Hoehe/Breite statt eines einzelnen linearen Positionszaehlers --
# v.a. Qwen2-VL/Qwen2.5-VL/Qwen3-VL-artige Vision-Modelle sowie deren reine
# Text-Varianten, die denselben RoPE-Typ beibehalten). Aeusserte sich hier
# als:
#   find_slot: non-consecutive token position N after M for sequence 0
#   ... for M-RoPE, it is required that the position satisfies: X < Y
# Ursache: llama-cpp-python's generischer Speculative-Decoding-Verify-Pfad
# (High-Level-Llama.generate/_create_completion) nimmt beim Verwerfen
# abgelehnter Kandidaten-Tokens einen einfachen, linearen Positionszaehler
# an (naechstes Token = letzte gespeicherte Position + 1). Bei M-RoPE ist
# die Position aber mehrdimensional und nicht einfach fortlaufend --
# dieselbe Grundannahme, die schon bei Hybrid-/SSM-Modellen (siehe oben)
# nicht gilt, nur mit anderer Fehlerursache (State nicht partiell
# zuruecknehmbar) und anderem Symptom (X < Y verletzt statt "partial kv
# removal not supported"). RecurrentStateCheckpoint (siehe dort) hilft hier
# NICHT, weil das Problem nicht in der KV-Ruecknahme selbst liegt, sondern
# in der Positionsberechnung nach jedem Verify-Schritt -- deshalb fuer
# M-RoPE-Modelle Speculative Decoding komplett deaktivieren, analog zur
# Hybrid-/SSM-Behandlung, statt eines Checkpoint-Workarounds.
MROPE_ARCH_MARKERS = (
    "qwen2vl", "qwen2-vl", "qwen2_vl",
    "qwen2.5vl", "qwen2.5-vl", "qwen2_5vl", "qwen2_5_vl",
    "qwen3vl", "qwen3-vl", "qwen3_vl",
)


def read_gguf_block_count(model_path):
    """Liest general.architecture UND {architecture}.block_count (Anzahl
    Transformer-/Hybrid-Layer) aus dem GGUF-Header, gleicher minimaler
    Parser-Mechanismus wie detect_gguf_architecture. Wird separat gehalten
    (nicht in detect_gguf_architecture eingebaut), weil block_count erst
    bekannt ist, NACHDEM general.architecture gelesen wurde (der Key-Name
    haengt von der Architektur ab, z.B. 'qwen35.block_count') -- der
    fruehe break in detect_gguf_architecture (sobald arch+name gefunden
    sind) wuerde block_count sonst u.U. verpassen, falls es im GGUF vor
    general.name aber nach general.architecture kommt oder umgekehrt.
    Gibt (block_count:int|None, architecture:str) zurueck."""
    try:
        with open(model_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None, ""
            f.read(4)  # version, hier nicht gebraucht

            def read_u64():
                return int.from_bytes(f.read(8), "little")

            def read_str():
                length = read_u64()
                return f.read(length).decode("utf-8", errors="replace")

            GGUF_TYPE_READERS_FIXED = {
                0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
            }

            def skip_value(vtype):
                if vtype == 8:
                    read_str()
                elif vtype == 9:
                    elem_type = int.from_bytes(f.read(4), "little")
                    count = read_u64()
                    for _ in range(count):
                        skip_value(elem_type)
                else:
                    size = GGUF_TYPE_READERS_FIXED.get(vtype)
                    if size is None:
                        raise ValueError(f"unbekannter GGUF value type {vtype}")
                    f.read(size)

            f.read(8)  # n_tensors, hier nicht gebraucht
            n_kv = read_u64()
            arch = ""
            block_count = None
            for _ in range(n_kv):
                key = read_str()
                vtype = int.from_bytes(f.read(4), "little")
                if key == "general.architecture" and vtype == 8:
                    arch = read_str()
                elif key.endswith(".block_count") and vtype in (4, 5):
                    # UINT32=4 / INT32=5, je nach GGUF-Schreiber
                    block_count = int.from_bytes(f.read(4), "little")
                    if vtype == 5 and block_count >= 2**31:
                        block_count -= 2**32
                else:
                    skip_value(vtype)
            return block_count, arch
    except Exception:
        return None, ""


def query_free_vram_mb():
    """Freies VRAM in MiB der ersten sichtbaren NVIDIA-GPU, per nvidia-smi
    (kein zusaetzliches Python-Paket noetig, auf jedem System mit NVIDIA-
    Treiber vorhanden). None bei Fehler/kein nvidia-smi im PATH/kein NVIDIA
    -- der Aufrufer muss das als 'Empfehlung nicht moeglich' behandeln,
    nicht als 0 interpretieren (0 wuerde faelschlich 'kein VRAM frei'
    suggerieren)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        first_line = out.stdout.strip().splitlines()[0].strip()
        return int(first_line)
    except Exception:
        return None


def estimate_optimal_gpu_layers(model_path, n_ctx, kv_cache_type, log_fn=None, is_hybrid_ssm=False):
    """Grobe Empfehlung fuer n_gpu_layers auf Basis von: Dateigroesse /
    block_count (Bytes pro Layer, Naeherung -- Attention- und FFN-Layer
    sind unterschiedlich gross, aber im Mittel brauchbar) und freiem VRAM
    per nvidia-smi. Reine INFO-Ausgabe im Log, AENDERT NICHTS automatisch
    an n_gpu_layers -- der Nutzer soll selbst entscheiden, nicht ungefragt
    uebersteuert werden.

    Sicherheitsmarge: reserviert ca. 1200 MiB fuer KV-Cache + Compute-
    Buffer + CUDA-Kontext-Overhead, weil diese NICHT in der reinen
    Gewichte-pro-Layer-Rechnung stecken. Bei sehr grossem n_ctx (siehe
    Aufrufer) faellt diese Marge zu klein aus -- deshalb ausdruecklich nur
    als Ausgangspunkt zum Ausprobieren gedacht, nicht als exakter Wert.

    is_hybrid_ssm: bei Hybrid-/SSM-Architekturen (siehe
    HYBRID_SSM_ARCH_MARKERS) ist der KV-Margin-Wert unten NICHT gemeint --
    deren rekurrenter State ist von n_ctx praktisch unabhaengig (siehe
    RecurrentStateCheckpoint-Docstring), die uebliche 'KV-Cache waechst
    mit dem Kontext'-Heuristik passt hier nicht. Bei True wird die feste,
    kleine Hybrid/SSM-State-Marge verwendet statt der n_ctx-abhaengigen."""
    log = log_fn or (lambda msg: None)
    block_count, arch = read_gguf_block_count(model_path)
    if not block_count or block_count <= 0:
        return  # kein block_count gefunden -- keine Empfehlung moeglich

    try:
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    except OSError:
        return
    bytes_per_layer_mb = file_size_mb / block_count

    free_vram_mb = query_free_vram_mb()
    if free_vram_mb is None:
        log(
            f"Hinweis: Modell hat {block_count} Layer "
            f"(~{bytes_per_layer_mb:.0f} MiB/Layer bei dieser Quantisierung). "
            "nvidia-smi nicht verfuegbar -- keine automatische "
            "n_gpu_layers-Empfehlung moeglich."
        )
        return

    # KV-Cache-Groesse haengt stark von n_ctx und kv_cache_type ab -- grobe
    # Reservierung, kein exakter Wert (siehe Docstring). q8_0/q4_0 KV
    # braucht deutlich weniger als f16, daher weniger Marge bei kleinerem
    # kv_cache_type. Reine Heuristik zur groben Ausrichtung, nicht mehr.
    # Ausnahme Hybrid-/SSM-Modelle: deren State ist fix und klein, unabhaengig
    # von n_ctx (siehe is_hybrid_ssm-Docstring oben) -- feste kleine Marge.
    if is_hybrid_ssm:
        kv_margin_mb = 200
    else:
        kv_margin_mb = 800 if kv_cache_type in ("q4_0", "q4_1", "q8_0") else 1600
    overhead_mb = 500  # CUDA-Kontext, Compute-Buffer, Fragmentierung
    usable_mb = max(free_vram_mb - kv_margin_mb - overhead_mb, 0)
    recommended = int(usable_mb / bytes_per_layer_mb) if bytes_per_layer_mb > 0 else 0
    recommended = max(0, min(recommended, block_count))

    log(
        f"Hinweis: Modell hat {block_count} Layer "
        f"(~{bytes_per_layer_mb:.0f} MiB/Layer), freies VRAM ~{free_vram_mb} MiB "
        f"(nvidia-smi). Grobe Empfehlung: n_gpu_layers ~{recommended} von "
        f"{block_count} (Reserve ~{kv_margin_mb + overhead_mb} MiB fuer "
        "KV-Cache/Compute-Buffer bei diesem n_ctx/kv_cache_type). Nur ein "
        "Ausgangspunkt zum Ausprobieren, kein exakter Wert -- n_gpu_layers "
        "wird dadurch NICHT automatisch geaendert."
    )


def try_enable_native_recurrent_rollback(llama_cpp_mod, n_rs_seq=4):
    """Aktiviert den NATIVEN partiellen Rollback fuer rekurrenten State
    (Mamba/RWKV/Hybrid-Linear-Attention wie Qwen3.5), falls die
    installierte llama-cpp-python-Version das C++-seitige n_rs_seq-Feature
    mitbringt (llama_context_params.n_rs_seq, [EXPERIMENTAL] in
    llama.cpp's include/llama.h -- 0 = aus, >0 = Anzahl Snapshot-Slots).

    WICHTIG -- ersetzt den alten RecurrentStateCheckpoint-Python-Workaround:
    llama.cpp's eigener llama_memory_recurrent::seq_rm() kann inzwischen
    (Stand der Vendor-Version, siehe src/llama-memory-recurrent.cpp)
    selbst einen begrenzten partiellen Rollback des rekurrenten State
    durchfuehren -- exakt das, was RecurrentStateCheckpoint bisher in
    Python nachgebaut hat (Snapshot per llama_state_seq_get_data() vor
    jedem Draft-Versuch, Restore bei Ablehnung). Der native Weg ist in
    jeder Hinsicht besser: kein Python-Overhead pro Draft-Versuch, keine
    fehleranfaellige Positions-Berechnung auf Python-Seite (siehe der
    Bugfix in _CheckpointingDraftModel.save() weiter unten -- diese ganze
    Fehlerklasse existiert beim nativen Weg gar nicht erst, weil llama.cpp
    selbst weiss, welche Position gerade aktuell ist), und funktioniert
    dadurch auch fuer den GANZ NORMALEN Praefix-Rollback (nicht nur fuer
    Speculative-Decoding-Ablehnungen), was der alte Checkpoint-Wrapper nie
    konnte, weil er nur beim Draft-Model-Aufruf einen Snapshot anlegte.

    Funktioniert per gezieltem Monkey-Patch auf
    llama_cpp.llama_context_default_params, weil llama-cpp-python's High-
    Level-Llama-Klasse dieses Feld nirgends selbst setzt (kein Kwarg dafuer
    vorhanden) und der llama_context bereits waehrend Llama.__init__ mit
    den zu dem Zeitpunkt aktuellen context_params gebaut wird -- ein
    nachtraegliches Setzen NACH der Konstruktion waere wirkungslos.

    Gibt True zurueck, wenn der Patch gesetzt wurde (inkl. eines
    Rueckgabe-Wrappers, der sich nach EINMALIGEM Gebrauch selbst wieder
    entfernt, damit andere/spaetere Llama-Instanzen -- z.B. ein zweites,
    nicht-hybrides Modell in derselben Sitzung -- nicht ungewollt
    denselben Wert erben). False, wenn n_rs_seq in dieser Version nicht
    existiert (aelteres llama-cpp-python) -- Aufrufer soll dann auf den
    alten RecurrentStateCheckpoint-Fallback zurueckfallen."""
    try:
        default_params_fn = llama_cpp_mod.llama_context_default_params
        # Feature-Detection: existiert das Feld im ctypes-Struct ueberhaupt?
        probe = default_params_fn()
        if not hasattr(probe, "n_rs_seq"):
            return False
    except Exception:
        return False

    def _patched_default_params():
        params = default_params_fn()
        params.n_rs_seq = n_rs_seq
        llama_cpp_mod.llama_context_default_params = default_params_fn  # einmalig, dann zurueck
        return params

    llama_cpp_mod.llama_context_default_params = _patched_default_params
    return True


def _patch_draft_model_scores_ringbuffer(llm, log_fn=None):
    """Behebt den 'could not broadcast input array from shape (N,) into
    shape (0,)'-Fehler in llama_decode bei aktivem Speculative Decoding
    (draft_model), OHNE den RAM-Bedarf auf n_ctx*n_vocab*4 Bytes
    hochzutreiben (das waeren bei grossem n_ctx/Vokabular leicht 40+ GiB,
    siehe Kommentar in LlamaCppEngine.load()).

    Hintergrund: llama-cpp-python's Llama.__init__ setzt
        self._logits_all = logits_all if draft_model is None else True
    sobald ein draft_model uebergeben wird -- unabhaengig vom rohen
    logits_all-Kwarg. eval() schreibt dann in JEDEM Fall mit dem absoluten,
    ueber die ganze Konversation hinweg wachsenden n_past als Zeilenindex
    in self.scores:
        self.scores[n_past : n_past + n_tokens, :].reshape(-1)[::] = logits
    Die GROESSE von self.scores wird aber nur anhand des rohen logits_all-
    Arguments entschieden (n_ctx Zeilen nur wenn logits_all=True beim
    Aufruf war, sonst n_batch Zeilen) -- das 'oder draft_model' aus der
    _logits_all-Zeile fliesst dort NICHT ein. Bleibt logits_all beim
    Aufruf False (wie in diesem Skript), ist der Puffer zu klein fuer die
    absolute Adressierung, sobald n_tokens > n_batch waechst.

    Da self.scores in diesem Skript nirgends GELESEN wird (kein
    logprobs=..., keine eigene stopping_criteria -- die einzigen beiden
    Konsumenten von self.scores/self._scores; Sampling und Speculative-
    Verify laufen komplett ueber den C-seitigen Logit-Puffer mit relativer
    Indizierung, nicht ueber self.scores), reicht ein kleiner Ringpuffer:
    dieselbe Zeilenzahl wie ohne Speculative Decoding (n_batch), aber mit
    Schreibindex modulo Puffergroesse statt absolutem n_past. Ersetzt dafuer
    die gebundene eval()-Methode dieser EINEN llm-Instanz durch eine
    ansonsten 1:1 identische Kopie (Stand llama_cpp_python 0.3.34); nur die
    Scores-Schreibzeile wurde geaendert. Betrifft NICHT die KV-Cache-/
    Token-Buchhaltung (self.input_ids, self.n_tokens, kv_cache_seq_rm) --
    die bleiben unveraendert absolut indiziert, wie es llama.cpp's
    eigentlicher (positions-adressierter) KV-Cache auch erwartet.

    ACHTUNG Versions-Abhaengigkeit: dieser Patch dupliziert die interne
    Struktur von Llama.eval(). Falls eine kuenftige llama-cpp-python-
    Version eval() strukturell aendert (neue Attribute, andere Batch-API),
    kann dieser Patch falsch werden oder mit AttributeError fehlschlagen --
    deshalb in einen try/except gehuellt, das bei JEDEM Fehler klar
    zurueckmeldet und die Original-eval()-Methode unangetastet laesst
    (dann tritt der urspruengliche Broadcast-Fehler wieder auf, statt
    still falsche Ergebnisse zu produzieren)."""
    import types

    import numpy as np

    original_eval = llm.eval

    try:
        buf_rows = llm.scores.shape[0]
        if buf_rows <= 0:
            raise ValueError(f"unerwartete Scores-Puffergroesse: {llm.scores.shape}")

        def _ringbuffer_eval(self, tokens):
            self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)
            for i in range(0, len(tokens), self.n_batch):
                batch = tokens[i: min(len(tokens), i + self.n_batch)]
                n_past = self.n_tokens
                n_tok = len(batch)
                self._batch.set_batch(
                    batch=batch, n_past=n_past, logits_all=self._logits_all
                )
                self._ctx.decode(self._batch)
                # Token-Buchhaltung bleibt absolut indiziert, unveraendert
                # gegenueber dem Original (input_ids ist n_ctx-gross und
                # dafuer auch dimensioniert).
                self.input_ids[n_past: n_past + n_tok] = batch
                if self._logits_all:
                    cols = self._n_vocab
                    logits = np.ctypeslib.as_array(
                        self._ctx.get_logits(), shape=(n_tok * cols,)
                    )
                    # NUR diese Zeile geaendert: modulo statt absolutem
                    # n_past, damit ein kleiner, fixer Puffer (buf_rows
                    # Zeilen) nie ueberlaufen kann, egal wie lang die
                    # Konversation wird. self.scores wird in diesem Skript
                    # nirgends zurueckgelesen, ein Ringpuffer ist daher
                    # unbedenklich (siehe Docstring von
                    # _patch_draft_model_scores_ringbuffer).
                    row0 = n_past % buf_rows
                    if row0 + n_tok <= buf_rows:
                        self.scores[row0: row0 + n_tok, :].reshape(-1)[::] = logits
                    else:
                        first = buf_rows - row0
                        self.scores[row0:buf_rows, :].reshape(-1)[::] = logits[: first * cols]
                        self.scores[0: n_tok - first, :].reshape(-1)[::] = logits[first * cols:]
                self.n_tokens += n_tok
                self._requires_eval = False

        llm.eval = types.MethodType(_ringbuffer_eval, llm)
        if log_fn:
            log_fn(
                "Hinweis: Scores-Puffer-Ringpuffer-Patch fuer Speculative "
                "Decoding aktiv (siehe _patch_draft_model_scores_ringbuffer) "
                f"-- Puffer bleibt bei {buf_rows} Zeilen statt n_ctx, "
                "verhindert den 'could not broadcast ... into shape (0,)'-"
                "Fehler ohne den RAM-Bedarf auf n_ctx*n_vocab*4 Bytes "
                "hochzutreiben."
            )
    except Exception as e:
        llm.eval = original_eval
        if log_fn:
            log_fn(
                f"WARNUNG: Scores-Ringpuffer-Patch konnte nicht angewendet "
                f"werden ({e}) -- eval() bleibt unveraendert. Bei laengeren "
                "Konversationen mit aktivem Speculative Decoding kann der "
                "'could not broadcast ... into shape (0,)'-Fehler dann "
                "wieder auftreten; im Zweifel Speculative Decoding fuer "
                "diese Sitzung deaktivieren."
            )


def detect_gguf_architecture(model_path):
    """Liest general.architecture (und general.name als Fallback) direkt aus
    dem GGUF-Header, ohne das Modell zu laden. Reines Parsen des GGUF-
    Binaerformats (Magic + Version + KV-Pairs) -- kein zusaetzliches Paket
    noetig, funktioniert unabhaengig von der llama-cpp-python-Version.
    Gibt (architecture_str, raw_name_str) zurueck, beides ggf. "" bei
    Parse-Fehlern/unbekanntem Format (dann besser vorsichtig sein statt
    einen Fehler zu werfen -- der Aufrufer behandelt "" als 'unbekannt')."""
    try:
        with open(model_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return "", ""
            version = int.from_bytes(f.read(4), "little")

            def read_u64():
                return int.from_bytes(f.read(8), "little")

            def read_str():
                length = read_u64()
                return f.read(length).decode("utf-8", errors="replace")

            GGUF_TYPE_READERS_FIXED = {
                0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
            }

            def skip_value(vtype):
                if vtype == 8:  # STRING
                    read_str()
                elif vtype == 9:  # ARRAY
                    elem_type = int.from_bytes(f.read(4), "little")
                    count = read_u64()
                    for _ in range(count):
                        skip_value(elem_type)
                else:
                    size = GGUF_TYPE_READERS_FIXED.get(vtype)
                    if size is None:
                        raise ValueError(f"unbekannter GGUF value type {vtype}")
                    f.read(size)

            n_tensors = read_u64()
            n_kv = read_u64()
            arch = ""
            name = ""
            for _ in range(n_kv):
                key = read_str()
                vtype = int.from_bytes(f.read(4), "little")
                if key == "general.architecture" and vtype == 8:
                    arch = read_str()
                elif key == "general.name" and vtype == 8:
                    name = read_str()
                else:
                    skip_value(vtype)
                if arch and name:
                    break
            return arch, name
    except Exception:
        # Kaputte/unerwartete Datei, altes GGUF-Format o.ae. -- lieber ""
        # zurueckgeben (Aufrufer behandelt das als 'unbekannt', nicht als
        # 'sicher kein Hybrid') als das Laden hart abzubrechen.
        return "", ""


def is_hybrid_ssm_architecture(model_path):
    """True, wenn Architektur ODER Dateiname auf einen Mamba-/SSM-/Hybrid-
    Linear-Attention-artigen State hindeuten (siehe HYBRID_SSM_ARCH_MARKERS).
    Prueft zusaetzlich den Dateinamen als Fallback, falls general.architecture
    im GGUF fehlt/leer ist oder ein noch unbekannter Architektur-String
    verwendet wird, das Modell aber z.B. "Qwen3.5" im Namen traegt."""
    arch, name = detect_gguf_architecture(model_path)
    haystack = f"{arch} {name} {os.path.basename(model_path)}".lower()
    return any(marker in haystack for marker in HYBRID_SSM_ARCH_MARKERS), arch, name


def is_mrope_architecture(model_path):
    """True, wenn Architektur ODER Dateiname auf ein M-RoPE-Modell
    hindeuten (siehe MROPE_ARCH_MARKERS-Docstring oben). Gleiches
    Erkennungsprinzip wie is_hybrid_ssm_architecture (Architektur-String
    bevorzugt, Dateiname als Fallback)."""
    arch, name = detect_gguf_architecture(model_path)
    haystack = f"{arch} {name} {os.path.basename(model_path)}".lower()
    return any(marker in haystack for marker in MROPE_ARCH_MARKERS), arch, name


class LlamaCppEngine:

    """Kapselt Laden und Streaming-Generierung ueber llama-cpp-python."""



    def __init__(self, log_fn):

        self.log = log_fn

        self.llm = None

        self.draft_model = None

        self.n_ctx = 4096

        self.is_hybrid_ssm = False
        self.is_mrope = False
        self._original_kv_cache_seq_rm = None
        self._native_rs_rollback = False



    def load(self, model_path, n_gpu_layers, n_ctx, n_threads, n_batch=512,
              offload_kqv=True, flash_attn=False, kv_cache_type="f16",
              n_ubatch=None, n_threads_batch=None,
              speculative_decoding=True, spec_num_pred_tokens=None,
              mtp_enabled=False, mtp_draft_n_max=3):
        """
        n_ubatch: physische Batch-Groesse fuers Prompt-Processing (Prefill).
            Getrennt von n_batch (logische Batch-Groesse): kleineres n_ubatch
            zerlegt einen grossen Prefill (z.B. nach einem KV-Cache-Miss durch
            Chunk-Eviction, siehe ChatStore.build_window) in kleinere Haeppchen
            -- reduziert den VRAM-Spitzenbedarf waehrend des Prompt-Processing,
            ohne die Fenstergroesse selbst zu aendern. None = wie n_batch.

        speculative_decoding: aktiviert llama.cpp's eingebautes Prompt-Lookup-
            Decoding (n-Gramm-basierte Selbstspekulation, kein zweites Modell
            noetig -- llama_cpp.llama_speculative.LlamaPromptLookupDecoding).
            Beschleunigt NICHT das Prompt-Processing (Prefill), sondern die
            Token-fuer-Token-Dekodierphase: pro Schritt werden mehrere
            Kandidat-Token aus zuvor gesehenen n-Grammen des Kontexts
            vorgeschlagen und in einem Batch gegen das Hauptmodell
            verifiziert, statt strikt ein Token nach dem anderen zu
            generieren. Greift besonders bei viel woertlicher Wiederholung
            aus dem Kontext (Code-Aenderungen, Zitate aus angehaengten
            Dateien, Diff-/Tool-Ausgaben) -- genau der Use-Case dieses Tools.
            Praktischer Ersatz fuer eine eigene "PAT"-Attention-Kernel-
            Implementierung (llama.cpp exponiert keinen Python-Hook fuer
            eigene CUDA-Attention-Kernel), zielt aber auf dieselbe Groesse
            (Tokens/Sekunde beim Dekodieren).

        Praefix-Wiederverwendung (Prompt-Processing/Prefill) laeuft NICHT
            mehr ueber einen expliziten set_cache()-Snapshot-Mechanismus
            (LlamaRAMCache/LlamaDiskCache) -- der wurde entfernt, weil er in
            Kombination mit Speculative Decoding zu einem M-RoPE-
            Positionsfehler fuehrte (siehe Kommentar weiter unten im Code).
            Stattdessen nutzt die App weiterhin die eingebaute Live-Praefix-
            Wiederverwendung von llama-cpp-python innerhalb des laufenden
            Prozesses, kombiniert mit _prepend_to_last_message (haengt
            volatile Zusatzinfos ans Ende statt an den Anfang), damit der
            Prompt-Anfang byte-stabil bleibt und wiederverwendet werden kann.

        speculative_decoding nutzt kein MTP mehr (dafuer gibt es fuer dieses
            Modell keine Variante, und es wuerde ohnehin mehr VRAM fuer die
            trainierten Zusatzkoepfe brauchen). Stattdessen kommt
            MultiPaperDraftModel zum Einsatz -- eine Kombination aus
            mehreren modellfreien Kandidaten-Generatoren (Suffix-Index ueber
            die gesamte Historie, turnuebergreifender Fortsetzungs-Cache,
            Token-Recycling-Fallback, adaptive Kandidatenlaenge). Siehe
            Docstring der Klasse fuer die einzelnen Papers und die
            Begruendung, welche NICHT aufgenommen wurden. Kein zusaetzliches
            VRAM, keine zweite GPU, kein zweites Modell.
        """
        from llama_cpp import Llama
        import llama_cpp as llama_cpp_mod

        # BUGFIX (llama-cpp-python, nicht architekturspezifisch, faellt aber
        # v.a. bei Speculative Decoding + krummer n_ctx auf): llama.cpp's
        # C++-Kern rundet n_ctx intern IMMER auf ein Vielfaches von 256 auf
        # (GGML_PAD(cparams.n_ctx, 256), siehe llama-context.cpp). self.
        # llm.n_ctx() fragt danach genau diesen AUFGERUNDETEN Wert ab (z.B.
        # 50176 bei angeforderten 50000). Der Speculative-Decoding-Guard in
        # llama-cpp-python's eigener generate()-Schleife
        # (llama_cpp/llama.py: "tokens.extend(draft_tokens[:self._n_ctx -
        # self.n_tokens - len(tokens)])") verwendet genau diesen
        # aufgerundeten Wert -- ABER self.scores/self.input_ids (die festen
        # NumPy-Puffer fuer Logits/Tokens) werden mit dem URSPRUENGLICHEN,
        # NICHT aufgerundeten n_ctx-Parameter allokiert (das passiert VOR
        # dem eigentlichen llama_new_context_with_model-Aufruf, mit dem
        # rohen Python-Wert). Bei nicht durch 256 teilbarem n_ctx (z.B.
        # 50000) erlaubt der Guard also bis zu 176 Tokens mehr als in den
        # Puffern tatsaechlich Platz ist -- sobald die Generierung nahe an
        # die angeforderte Grenze heranwaechst, kann Speculative Decoding
        # dann mehr Kandidat-Tokens vorschlagen als der Puffer fasst:
        #   ValueError: could not broadcast input array from shape (N,)
        #   into shape (0,)
        # Fix: n_ctx hier selbst auf ein Vielfaches von 256 aufrunden,
        # BEVOR es an Llama() uebergeben wird -- dann sind aufgerundeter
        # und roher Wert identisch, die Diskrepanz kann nicht mehr
        # entstehen. Aendert die tatsaechlich nutzbare Kontextgroesse nur
        # nach oben (nie nach unten), ist also unkritisch fuer bestehende
        # Konversationen/Fenstergroessen.
        _requested_n_ctx = n_ctx
        n_ctx = ((n_ctx + 255) // 256) * 256
        if n_ctx != _requested_n_ctx:
            self.log(
                f"Hinweis: n_ctx {_requested_n_ctx} ist nicht durch 256 "
                f"teilbar -- auf {n_ctx} aufgerundet (llama.cpp rundet "
                "intern ohnehin auf, aber ohne diesen Fix bleiben "
                "llama-cpp-python's eigene Logit-/Token-Puffer bei der "
                "kleineren Zahl -- das fuehrte bei Speculative Decoding "
                "nahe der Kontextgrenze zu einem Broadcast-Fehler)."
            )

        # Architektur-Check VOR dem eigentlichen Laden: bei Mamba-/SSM-/
        # Hybrid-Linear-Attention-Modellen (z.B. Qwen3.5) ist Speculative
        # Decoding strukturell kaputt (siehe HYBRID_SSM_ARCH_MARKERS oben)
        # -- also hier automatisch abschalten, statt erst nach einem
        # llama_decode-Crash reaktiv zu reagieren (der alte Retry in
        # _handle_decode_failure greift zwar auch, aber eben erst NACHDEM
        # der Fehler schon aufgetreten ist).
        self.is_hybrid_ssm, detected_arch, detected_name = is_hybrid_ssm_architecture(model_path)
        self._native_rs_rollback = False
        if self.is_hybrid_ssm and speculative_decoding:
            self._native_rs_rollback = try_enable_native_recurrent_rollback(llama_cpp_mod)
            if self._native_rs_rollback:
                self.log(
                    f"Erkannt: Hybrid-/SSM-artige Architektur (architecture="
                    f"'{detected_arch or '?'}', name='{detected_name or '?'}'). "
                    "Diese llama-cpp-python-Version unterstuetzt nativen "
                    "partiellen Rollback fuer rekurrenten State (n_rs_seq, "
                    "[EXPERIMENTAL] in llama.cpp) -- aktiviert. Kein "
                    "Python-seitiger Checkpoint-Workaround noetig, "
                    "Speculative Decoding UND normaler Praefix-Rollback "
                    "laufen jetzt beide ueber den nativen Mechanismus."
                )
            else:
                self.log(
                    f"Erkannt: Hybrid-/SSM-artige Architektur (architecture="
                    f"'{detected_arch or '?'}', name='{detected_name or '?'}'). "
                    "Der normale KV-Rollback (llama_memory_seq_rm) geht bei "
                    "dieser Architektur nicht (State nicht partiell "
                    "zuruecknehmbar) -- diese llama-cpp-python-Version hat "
                    "noch keinen nativen n_rs_seq-Rollback, versuche "
                    "stattdessen Checkpoint-basiertes Speculative Decoding "
                    "(RecurrentStateCheckpoint, siehe dort) als Fallback. "
                    "Faellt automatisch auf Deaktivierung zurueck, falls "
                    "auch dieser Mechanismus nicht verfuegbar ist."
                )

        # Zweiter, unabhaengiger Architektur-Check: M-RoPE-Modelle (siehe
        # MROPE_ARCH_MARKERS-Docstring oben). Anders als beim Hybrid-/SSM-
        # Fall gibt es hier KEINEN Checkpoint-Workaround -- das Problem
        # liegt in der Positionsberechnung selbst (X < Y verletzt), nicht
        # in der KV-Ruecknahme. Speculative Decoding bleibt bei erkanntem
        # M-RoPE-Modell also komplett aus, unabhaengig davon, ob
        # RecurrentStateCheckpoint verfuegbar waere.
        self.is_mrope, mrope_arch, mrope_name = is_mrope_architecture(model_path)
        if self.is_mrope and speculative_decoding:
            self.log(
                f"Erkannt: M-RoPE-Architektur (architecture="
                f"'{mrope_arch or '?'}', name='{mrope_name or '?'}'). "
                "Speculative Decoding (MultiPaperDraftModel) wird fuer "
                "dieses Modell automatisch deaktiviert: der generische "
                "Verify-Pfad nimmt einen linearen Positionszaehler an, "
                "M-RoPE-Positionen sind aber mehrdimensional -- das fuehrt "
                "zu 'non-consecutive token position' / 'X < Y' Fehlern in "
                "llama_decode. Kein Workaround dafuer verfuegbar (anders "
                "als beim Hybrid-/SSM-Fall oben), Prefix-Caching ueber "
                "_prepend_to_last_message bleibt aber weiterhin aktiv."
            )



        # KV-Cache-Datentyp: kleinerer Typ = kleinerer KV-Cache-Puffer.

        # Quantisierter KV-Cache (Q8_0/Q4_0) braucht in aktuellem llama.cpp

        # i.d.R. Flash-Attention -- wird hier automatisch mit angeschaltet.

        # Hinweis: GGML kennt in Mainline-llama.cpp keinen echten 6-Bit-Typ
        # fuer den KV-Cache. Unterstuetzt sind offiziell nur f16, q8_0
        # (8-bit), q5_1/q5_0 (5-bit) und q4_1/q4_0 (4-bit) -- die K/V-
        # Dequant-Kernels fuer Q6_K existieren im ggml-Standardpaket nicht.
        # "q6_k" ist hier nur als optionaler Eintrag vorgesehen: er greift
        # NUR, wenn GGML_TYPE_Q6_K tatsaechlich im installierten llama_cpp-
        # Modul existiert (z.B. bei einem selbst kompilierten Fork mit
        # 6-bit-KV-Support, etwa "TurboQuant"-Forks von llama.cpp -- das ist
        # NICHT im normalen `pip install llama-cpp-python`-Wheel enthalten).
        # Ist das Attribut nicht vorhanden, faellt der bestehende Fallback
        # unten (kv_cache_type != "f16" und type_k is None) sauber auf F16
        # zurueck, statt beim Laden/Generieren mit GGML_ASSERT abzustuerzen.
        kv_type_map = {

            "f16": getattr(llama_cpp_mod, "GGML_TYPE_F16", None),

            "q8_0": getattr(llama_cpp_mod, "GGML_TYPE_Q8_0", None),

            "q6_k": getattr(llama_cpp_mod, "GGML_TYPE_Q6_K", None),

            "q5_1": getattr(llama_cpp_mod, "GGML_TYPE_Q5_1", None),

            "q5_0": getattr(llama_cpp_mod, "GGML_TYPE_Q5_0", None),

            "q4_1": getattr(llama_cpp_mod, "GGML_TYPE_Q4_1", None),

            "q4_0": getattr(llama_cpp_mod, "GGML_TYPE_Q4_0", None),

        }

        type_k = type_v = kv_type_map.get(kv_cache_type)

        if kv_cache_type != "f16" and type_k is not None:

            flash_attn = True



        self.log(f"Lade GGUF: {model_path}")

        # Reine INFO-Empfehlung, aendert n_gpu_layers NICHT automatisch
        # (siehe estimate_optimal_gpu_layers-Docstring). Nach der Angabe
        # "VRAM/RAM gehen gerade so" bewusst als Hinweis statt Auto-Tuning,
        # damit der Nutzer selbst entscheidet, ob er die Zahl ausreizt.
        estimate_optimal_gpu_layers(
            model_path, n_ctx, kv_cache_type, log_fn=self.log,
            is_hybrid_ssm=self.is_hybrid_ssm,
        )

        self.log(

            f"n_gpu_layers={n_gpu_layers}  n_ctx={n_ctx}  n_threads={n_threads}  "

            f"n_batch={n_batch}  offload_kqv={offload_kqv}  flash_attn={flash_attn}  "

            f"kv_cache_type={kv_cache_type}"

        )

        self.log(

            "Hinweis: bei n_gpu_layers > 0 ohne CUDA-Build von llama-cpp-python "

            "wird trotzdem alles auf CPU laufen (kein Fehler, nur langsamer)."

        )



        kwargs = dict(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_ubatch=n_ubatch if n_ubatch else n_batch,
            n_threads=n_threads if n_threads > 0 else None,
            n_threads_batch=n_threads_batch if n_threads_batch else None,
            offload_kqv=offload_kqv,
            flash_attn=flash_attn,
            verbose=True,
        )
        if type_k is not None and type_v is not None:
            kwargs["type_k"] = type_k
            kwargs["type_v"] = type_v
        elif kv_cache_type != "f16":
            self.log(
                f"Hinweis: KV-Cache-Typ '{kv_cache_type}' ist in dieser "
                "llama-cpp-python-Version nicht verfuegbar, bleibe bei F16. "
                + ("(q6_k braucht einen selbst kompilierten llama.cpp-Fork "
                   "mit 6-bit-KV-Support -- im normalen PyPI-Wheel gibt es "
                   "das nicht.)" if kv_cache_type == "q6_k" else "")
            )

        # MTP (Multi-Token Prediction, seit llama.cpp-PR #22673 / 16.05.2026
        # in Mainline): das GGUF traegt trainierte Zusatzkoepfe, die pro
        # Forward-Pass gleich mehrere zukuenftige Token vorschlagen, die dann
        # in EINEM Schritt gegen das Hauptmodell verifiziert werden -- kein
        # zweites Modell noetig (anders als beim draft_model-Pfad oben).
        # WICHTIG: Stand jetzt ist MTP nur ueber die C++-Binaries
        # llama-server/llama-cli verdrahtet (--spec-type draft-mtp
        # --spec-draft-n-max N). Die Python-Low-Level-Bindings von
        # llama-cpp-python kennen den Parameter noch nicht (bestaetigt u.a.
        # durch offene Upstream-Issues dazu) -- ein hartcodierter Kwarg wie
        # mtp=True wuerde also nur mit TypeError crashen. Deshalb hier ein
        # Feature-Detect ueber inspect.signature: greift automatisch, sobald
        # eine kuenftige llama-cpp-python-Version den Parameter nachliefert,
        # sonst sauberer Fallback auf normale (Nicht-MTP-)Generierung.
        self.mtp_active = False
        if mtp_enabled:
            import inspect
            try:
                llama_init_params = inspect.signature(Llama.__init__).parameters
            except (TypeError, ValueError):
                llama_init_params = {}
            mtp_kwarg_candidates = ("spec_type", "mtp", "draft_mtp", "spec_draft_n_max")
            found = [name for name in mtp_kwarg_candidates if name in llama_init_params]
            if "spec_type" in found:
                kwargs["spec_type"] = "mtp" if "draft_mtp" not in found else "draft-mtp"
                if "spec_draft_n_max" in found:
                    kwargs["spec_draft_n_max"] = mtp_draft_n_max
                self.mtp_active = True
                self.log(f"MTP aktiviert ueber llama-cpp-python (spec_draft_n_max={mtp_draft_n_max}).")
            elif "mtp" in found:
                kwargs["mtp"] = True
                if "spec_draft_n_max" in found:
                    kwargs["spec_draft_n_max"] = mtp_draft_n_max
                self.mtp_active = True
                self.log(f"MTP aktiviert ueber llama-cpp-python (spec_draft_n_max={mtp_draft_n_max}).")
            else:
                self.log(
                    "Hinweis: MTP angehakt, aber diese llama-cpp-python-"
                    "Version kennt noch keinen spec_type/mtp-Parameter im "
                    "Llama()-Konstruktor -- das MTP-GGUF laedt trotzdem "
                    "ganz normal (die Zusatzkoepfe werden nur ungenutzt "
                    "mitgeladen), es gibt aber KEINEN Speedup. Fuer echte "
                    "MTP-Beschleunigung aktuell nur ueber llama-server/"
                    "llama-cli moeglich (--spec-type draft-mtp "
                    "--spec-draft-n-max N), noch nicht ueber dieses Skript. "
                    "Pruefe ggf. auf ein Update von llama-cpp-python."
                )

        # Speculative Decoding: beschleunigt NICHT das Prompt-Processing
        # (Prefix-Cache oben), sondern die Token-fuer-Token-Dekodierphase.
        # Frueher stand hier ein MTP-Zweig (Multi-Token-Prediction ueber
        # trainierte Zusatzkoepfe) -- entfernt, weil (a) es fuer dieses
        # Modell keine MTP-GGUF-Variante gibt und (b) MTP-Zusatzkoepfe
        # ohnehin mehr VRAM brauchen wuerden, was hier ausdruecklich nicht
        # gewuenscht ist. Stattdessen: MultiPaperDraftModel (siehe
        # Klassen-Docstring oben fuer alle kombinierten Papers) -- rein
        # CPU/RAM-seitig, kein zusaetzliches VRAM, keine zweite GPU/kein
        # zweites Modell, funktioniert mit JEDEM geladenen GGUF.
        self.draft_model = None
        if speculative_decoding and self.is_mrope:
            speculative_decoding = False  # siehe M-RoPE-Check oben
        if speculative_decoding:
            self.draft_model = MultiPaperDraftModel(
                num_pred_tokens=spec_num_pred_tokens or (10 if n_gpu_layers != 0 else 2),
                log_fn=self.log,
            )
            kwargs["draft_model"] = self.draft_model
            self.log(
                "Speculative Decoding aktiv: MultiPaperDraftModel "
                "(Suffix-Index + Fortsetzungs-Cache + Token-Recycling + "
                "adaptive Kandidatenlaenge, kein Zusatz-VRAM)."
            )

        # WICHTIG -- eigentliche Ursache des 'could not broadcast input array
        # from shape (N,) into shape (0,)'-Fehlers in llama_decode:
        # llama-cpp-python's Llama.__init__ setzt intern
        #   self._logits_all = logits_all if draft_model is None else True
        # d.h. SOBALD ein draft_model uebergeben wird -- UNABHAENGIG davon,
        # was wir selbst fuer logits_all uebergeben -- verhaelt sich eval()
        # innerhalb der Bibliothek so, als waere logits_all=True, und
        # schreibt dort mit dem absoluten, ueber die ganze Konversation
        # hinweg wachsenden n_past/self.n_tokens als Zeilenindex in
        # self.scores. Die GROESSE von self.scores wird aber an anderer
        # Stelle im selben __init__ (np.ndarray(...)) nur anhand des ROHEN
        # logits_all-Arguments entschieden ((n_ctx, n_vocab) falls beim
        # Aufruf logits_all==True, sonst nur (n_batch, n_vocab)) -- ohne das
        # oben erwaehnte 'oder draft_model' zu beruecksichtigen. Wird
        # logits_all beim Aufruf nicht explizit gesetzt (Default False, wie
        # bisher in diesem Skript), bleibt der Puffer bei n_batch Zeilen,
        # waehrend eval() trotzdem mit dem absoluten, unbegrenzt wachsenden
        # n_past hineinschreibt. Sobald self.n_tokens > n_batch waechst (in
        # jeder laenger laufenden Konversation mit aktivem Draft-Modell
        # unvermeidlich), ist die Ziel-Slice leer -> exakt der beobachtete
        # Broadcast-Fehler. Das erklaert auch, warum weder Deaktivieren von
        # Speculative Decoding noch ein harter Kontext-Reset in
        # _handle_decode_failure dauerhaft hilft (siehe dortige Kommentare).
        #
        # NAHELIEGENDER, aber FALSCHER Fix: logits_all=True explizit an
        # Llama() uebergeben, damit der Puffer korrekt mit n_ctx statt
        # n_batch Zeilen alloziert wird. Das ist bei kleinem n_ctx/Vokabular
        # ok, aber bei grossem Vokabular (z.B. 248320 bei diesem Qwen3.5-
        # GGUF) und grossem n_ctx (hier 50176) waeren das 50176*248320*4
        # Bytes = 46.4 GiB -- schlicht nicht allozierbar (numpy.core.
        # _exceptions._ArrayMemoryError beim Laden).
        #
        # Tatsaechlich noetig ist so viel Speicher aber gar nicht: self.
        # scores wird in diesem Skript NIRGENDS gelesen (kein logprobs=...,
        # keine eigene stopping_criteria -- das sind die einzigen beiden
        # Konsumenten von self.scores/self._scores in llama-cpp-python; das
        # eigentliche Token-Sampling UND die Speculative-Decoding-
        # Verifikation lesen direkt aus dem C-seitigen Logit-Puffer via
        # self._sampler.sample(ctx, relativer_index) -- relativ zum
        # aktuellen n_tokens, NIE aus self.scores). self.scores ist bei uns
        # also ein reiner Schreib-nie-gelesen-Puffer -- ein Ringpuffer mit
        # fixer, kleiner Groesse (n_batch Zeilen, exakt wie ohne Speculative
        # Decoding auch) reicht dafuer vollkommen aus, wenn man beim
        # Schreiben modulo statt absolut indiziert. Deshalb hier NICHT
        # logits_all=True setzen (RAM-Explosion bei grossem n_ctx/Vokabular),
        # sondern nach dem Llama(**kwargs)-Aufbau die Schreib-Indizierung in
        # eval() selbst patchen -- siehe _patch_draft_model_scores_ringbuffer()
        # weiter oben, angewendet weiter unten sobald draft_model aktiv ist.

        try:
            self.llm = Llama(**kwargs)
        except TypeError as e:
            # Aeltere llama-cpp-python-Versionen kennen offload_kqv/flash_attn/
            # type_k/type_v/n_ubatch/n_threads_batch/draft_model ggf. noch
            # nicht -> schrittweise ohne diese Parameter erneut versuchen
            # statt ganz abzubrechen.
            self.log(f"Hinweis: Parameter wird von dieser llama-cpp-python-Version nicht unterstuetzt ({e}), versuche ohne erneut.")
            for key in ("offload_kqv", "flash_attn", "type_k", "type_v",
                        "n_ubatch", "n_threads_batch", "draft_model"):
                kwargs.pop(key, None)
            try:
                self.llm = Llama(**kwargs)
            except ValueError as e2:
                raise ValueError(self._diagnose_context_failure(n_ctx, kv_cache_type)) from e2
            if self.draft_model is not None:
                self.log(
                    "Hinweis: draft_model wurde beim Retry ohne diesen "
                    "Parameter entfernt -- Speculative Decoding ist fuer "
                    "diese llama-cpp-python-Version generell nicht aktiv, "
                    "unabhaengig von Hybrid-/SSM-Checkpointing."
                )
                self.draft_model = None
        except ValueError as e:
            # "Failed to create llama_context" kommt von llama.cpp praktisch
            # immer daher, dass der KV-Cache fuer das gewaehlte n_ctx nicht
            # allokiert werden konnte (RAM/VRAM reicht nicht) -- nicht von
            # einem falschen Parameter. Klartext statt nacktem Traceback.
            raise ValueError(self._diagnose_context_failure(n_ctx, kv_cache_type)) from e

        # Ringpuffer-Patch fuer den Scores-Puffer anwenden (siehe grosser
        # Kommentar oben) -- NUR wenn tatsaechlich noch ein Draft-Modell
        # aktiv ist (kann durch den TypeError-Retry-Block direkt darueber
        # bereits wieder auf None gesetzt worden sein, siehe dort).
        if self.draft_model is not None:
            _patch_draft_model_scores_ringbuffer(self.llm, self.log)

        # Checkpoint-basiertes Speculative Decoding fuer Hybrid-/SSM-Modelle
        # aktivieren (siehe RecurrentStateCheckpoint-Docstring oben). Muss
        # NACH dem Llama(**kwargs)-Aufbau passieren, weil self.llm._ctx erst
        # dann existiert -- draft_model selbst wurde schon beim Aufbau
        # gesetzt, wird hier aber per Instanz-Attribut nachtraeglich gegen
        # den Checkpoint-Wrapper ausgetauscht (generate() liest self.
        # draft_model bei jedem Aufruf frisch, kein Rebuild noetig).
        #
        # NUR als Fallback: wenn try_enable_native_recurrent_rollback()
        # oben schon gegriffen hat (self._native_rs_rollback == True),
        # macht llama.cpp den partiellen Rollback bereits selbst nativ und
        # BESSER (siehe try_enable_native_recurrent_rollback-Docstring) --
        # dieser ganze Python-Patch-Block wird dann komplett uebersprungen,
        # damit kein redundanter/widerspruechlicher zweiter Snapshot-
        # Mechanismus gleichzeitig auf denselben State zugreift.
        self._original_kv_cache_seq_rm = None  # siehe _handle_decode_failure
        if self._native_rs_rollback:
            pass  # nativer Mechanismus übernimmt, kein Python-Patch noetig
        elif self.is_hybrid_ssm and self.draft_model is not None:
            checkpoint = RecurrentStateCheckpoint(self.llm, log_fn=self.log)
            if checkpoint.supported:
                original_rm = self.llm._ctx.kv_cache_seq_rm
                self._original_kv_cache_seq_rm = original_rm

                def _patched_seq_rm(seq_id, p0, p1, _orig=original_rm, _ckpt=checkpoint):
                    if _ckpt.try_restore(p0):
                        return True
                    return _orig(seq_id, p0, p1)

                self.llm._ctx.kv_cache_seq_rm = _patched_seq_rm
                self.llm.draft_model = _CheckpointingDraftModel(self.draft_model, checkpoint, self.llm)
                self._recurrent_checkpoint = checkpoint
                self.log(
                    "Checkpoint-basiertes Speculative Decoding aktiv fuer "
                    "Hybrid-/SSM-Modell (kein Zusatz-VRAM, State-Snapshot-"
                    "Groesse ist fix, waechst nicht mit der Konversation)."
                )
            else:
                self.log(
                    "Deaktiviere Speculative Decoding fuer dieses Hybrid-/"
                    "SSM-Modell (Checkpoint-Mechanismus nicht verfuegbar, "
                    "siehe Hinweis oben)."
                )
                self.llm.draft_model = None
                self.draft_model = None

        # HINWEIS (ehemals hier: LlamaRAMCache/LlamaDiskCache via set_cache):
        # Diese eingebaute Save/Load-State-Cache-Implementierung von
        # llama-cpp-python wurde ENTFERNT. Sie hasht Prompt-Tokens und
        # ersetzt bei einem Treffer den laufenden KV-Cache per load_state()
        # durch einen zuvor per save_state() gesicherten Snapshot -- das ist
        # etwas ANDERES als die eingebaute Live-Praefix-Wiederverwendung
        # innerhalb EINES laufenden Kontexts (siehe _prepend_to_last_message
        # oben, die genau darauf ausgelegt ist). Das Snapshot-Restore hat den
        # Positionszaehler des wiederhergestellten Zustands nicht konsistent
        # mit dem neuen Prompt gehalten -- insbesondere im Zusammenspiel mit
        # Speculative Decoding, das direkt Decode-Batches manipuliert. Folge
        # war ein M-RoPE-Fehler ("the last position stored ... X = 138" vs.
        # "starting position ... Y = 129", verletzt die Bedingung X < Y),
        # llama_decode schlug fehl, und llama-cpp-python fing das zwar ab
        # ("cache miss", "prefix-match found but partial kv removal not
        # supported, re-evaluating full prompt"), aber genau dieses volle
        # Neu-Prozessieren ist der teure Fall, den der Cache eigentlich
        # vermeiden sollte -- er kostete hier also mehr, als er sparte. Die
        # Praefix-Wiederverwendung innerhalb einer laufenden Sitzung
        # funktioniert weiterhin automatisch ueber llama.cpp selbst, solange
        # der Prompt-Anfang byte-stabil bleibt (siehe _prepend_to_last_message
        # / _inject_effort_prompt / _inject_memory / _inject_tools_prompt).
        self.n_ctx = n_ctx
        scores_buf = getattr(self.llm, "scores", None)
        if scores_buf is not None:
            self.log(
                f"Scores-Puffer: shape={scores_buf.shape}, "
                f"{scores_buf.nbytes / 1e9:.2f} GB RAM "
                f"({'n_batch Zeilen + Ringpuffer-Patch aktiv' if self.draft_model is not None else 'n_batch Zeilen, wie ueblich ohne Speculative Decoding'})."
            )
        self.log("Modell geladen. Bereit fuer Generierung.")



    @staticmethod

    def _diagnose_context_failure(n_ctx, kv_cache_type):

        suggestions = []

        if n_ctx > 32768:

            suggestions.append(f"n_ctx von {n_ctx} deutlich senken (z.B. 8192 oder 16384 zum Testen)")

        if kv_cache_type == "f16":

            suggestions.append("KV-Cache-Praezision auf q8_0 oder q4_0 stellen (4-8x kleinerer KV-Cache)")

        suggestions.append("n_gpu_layers senken, falls VRAM (nicht RAM) der Engpass ist")

        bullet_list = "\n".join(f"  - {s}" for s in suggestions)

        return (

            "Kontext-Erstellung fehlgeschlagen (llama.cpp: 'Failed to create "

            "llama_context'). Das bedeutet fast immer: der KV-Cache fuer "

            f"n_ctx={n_ctx} passt nicht in den verfuegbaren RAM/VRAM -- der "

            "Speicherbedarf des KV-Cache waechst linear mit n_ctx und kann "

            "bei sehr grossen Werten schnell zwei- bis dreistellige GB "

            "erreichen, unabhaengig von der Modellgroesse selbst.\n"

            f"Vorschlaege:\n{bullet_list}"

        )



    def tokenize_count(self, text):

        """Exakte Tokenanzahl ueber den echten Modell-Tokenizer, mit grober

        Schaetzung als Fallback falls das Modell noch nicht laeuft."""

        if not text:

            return 0

        if self.llm is not None:

            try:

                return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

            except Exception:

                pass

        return max(1, len(text) // 3)  # grobe Schaetzung ohne geladenes Modell



    def _hard_reset_hybrid_context(self):
        """Setzt bei Hybrid-/SSM-Modellen den TATSAECHLICHEN Zustand
        zurueck -- nicht nur self.llm.n_tokens. Das eingebaute
        self.llm.reset() von llama-cpp-python setzt NUR den Python-
        seitigen Zaehler self.n_tokens auf 0 und geht davon aus, dass neu
        decodierte Tokens ab Position 0 den alten Zustand implizit
        ueberschreiben. Das stimmt fuer normalen Transformer-KV-Cache
        (per Position adressiert), NICHT aber fuer rekurrenten/SSM-State
        (Mamba/Hybrid-Linear-Attention wie Qwen3.5): das ist ein einzelner,
        sich fortlaufend aktualisierender Zustand pro Sequenz, nicht per
        Position adressierbar. Bleibt der native State von der letzten
        Runde stehen, kann v.a. in Kombination mit dem experimentellen
        n_rs_seq-Rollback (siehe try_enable_native_recurrent_rollback) die
        Python-seitige Buchhaltung (self.scores/self.n_tokens) aus dem
        Tritt geraten. Symptom: 'could not broadcast input array from
        shape (N,) into shape (0,)' beim naechsten llama_decode -- auch
        wenn Speculative Decoding laengst deaktiviert ist (siehe
        _handle_decode_failure).

        Probiert mehrere API-Varianten durch, weil sich die Namen
        zwischen llama.cpp-/llama-cpp-python-Versionen unterscheiden
        (aelter: kv_cache_*, neuer nach dem KV-Cache->Memory-Rename in
        llama.cpp: memory_*/llama_memory_clear). Setzt am Ende IMMER auch
        den Python-Zaehler zurueck, unabhaengig davon, ob ein nativer
        Clear-Aufruf gegriffen hat -- sonst schreibt der naechste eval()
        so oder so auf Basis eines falschen n_past in eine falsche Slice.
        Gibt True zurueck, wenn ein nativer Clear-Mechanismus gefunden
        und aufgerufen wurde, sonst False (dann half nur der Python-
        Zaehler-Reset, was bei dieser Architektur nicht ausreicht)."""
        if self.llm is None:
            return False
        import llama_cpp as llama_cpp_mod

        ctx = getattr(self.llm, "_ctx", None)
        cleared = False

        # Variante 1: Methode direkt auf dem High-Level-Context-Wrapper.
        for method_name in ("kv_cache_clear", "memory_clear"):
            fn = getattr(ctx, method_name, None)
            if callable(fn):
                try:
                    fn()
                    cleared = True
                    break
                except Exception:
                    pass

        # Variante 2: rohe ctypes-Funktionen direkt auf dem llama_context-
        # Pointer, unabhaengig vom High-Level-Wrapper.
        if not cleared:
            raw_ctx = getattr(ctx, "ctx", None) or getattr(ctx, "context", None)
            if raw_ctx is not None:
                fn = getattr(llama_cpp_mod, "llama_kv_cache_clear", None)
                if callable(fn):
                    try:
                        fn(raw_ctx)
                        cleared = True
                    except Exception:
                        pass
                if not cleared:
                    get_mem = getattr(llama_cpp_mod, "llama_get_memory", None)
                    mem_clear = getattr(llama_cpp_mod, "llama_memory_clear", None)
                    if callable(get_mem) and callable(mem_clear):
                        try:
                            mem = get_mem(raw_ctx)
                            mem_clear(mem, True)
                            cleared = True
                        except Exception:
                            pass

        try:
            self.llm.n_tokens = 0
        except Exception:
            pass
        try:
            self.llm.reset()
        except Exception:
            pass

        if cleared:
            self.log("Hinweis: Nativer KV-/rekurrenter State hart zurueckgesetzt.")
        else:
            self.log(
                "WARNUNG: Konnte den nativen KV-/rekurrenten State nicht "
                "hart zuruecksetzen (keine bekannte API in dieser "
                "llama-cpp-python-Version gefunden) -- nur der Python-"
                "Zaehler wurde zurueckgesetzt. Bei Hybrid-/SSM-Modellen "
                "kann das weiterhin zu Fehlern fuehren; im Zweifel Modell "
                "neu laden."
            )
        return cleared

    def _handle_decode_failure(self, exc, messages, max_new_tokens, temperature, top_p,
                                callback, reasoning_callback, tools, tool_choice,
                                max_reasoning_seconds, repeat_penalty, stop_event,
                                already_retried, hard_reset_retried=False):
        """Zentrale Behandlung von 'RuntimeError: llama_decode returned -1'.

        Haeufigste reale Ursache in diesem Skript: das custom
        MultiPaperDraftModel (Speculative Decoding) schlaegt dem Haupt-
        modell Kandidaten-Tokens vor, die llama.cpp intern nicht sauber
        verarbeiten kann (Zusammenspiel aus Draft-Model-Hook + KV-Cache-
        Handling ist in llama-cpp-python deutlich fragiler als normales
        Decoding ohne Draft-Modell). Statt den ganzen Worker-Thread mit
        einem Traceback abstuerzen zu lassen: einmalig ohne Speculative
        Decoding neu generieren. Bleibt der Fehler bestehen, liegt die
        Ursache woanders (z.B. n_ctx/VRAM) -- dann wird sauber ein
        RuntimeError mit Diagnose-Text nach oben gereicht, statt den
        Rohtraceback von llama-cpp-python zu zeigen.
        """
        self.log(f"FEHLER bei llama_decode: {exc}")

        if not already_retried and getattr(self.llm, "draft_model", None) is not None:
            hint = ""
            if "M-RoPE" in str(exc) or "non-consecutive token position" in str(exc):
                hint = (
                    " (Fehlertext deutet auf M-RoPE hin -- siehe "
                    "MROPE_ARCH_MARKERS: falls dieses Modell dort noch "
                    "nicht gelistet ist, bitte Architektur/Dateiname "
                    "ergaenzen, damit Speculative Decoding kuenftig gleich "
                    "beim Laden automatisch deaktiviert wird statt erst "
                    "nach diesem Fehler.)"
                )
            self.log(
                "Deaktiviere Speculative Decoding (MultiPaperDraftModel) fuer "
                "den Rest dieser Sitzung und generiere die Anfrage einmal neu -- "
                "das Draft-Modell ist die wahrscheinlichste Fehlerquelle fuer "
                "'llama_decode returned -1'. Bei Bedarf Modell neu laden, um "
                "Speculative Decoding wieder zu aktivieren." + hint
            )
            self.llm.draft_model = None
            self.draft_model = None
            # WICHTIG: bei Hybrid-/SSM-Modellen wurde zusaetzlich
            # self.llm._ctx.kv_cache_seq_rm auf _patched_seq_rm umgebogen
            # (siehe load()). Das alleinige Leeren von draft_model reicht
            # hier NICHT -- llama-cpp-python's generate()-Schleife ruft
            # kv_cache_seq_rm auch OHNE Draft-Modell auf (z.B. bei
            # normalem Praefix-Rollback ueber _prepend_to_last_message).
            # Ohne aktives Draft-Modell wird nie mehr ein Checkpoint per
            # RecurrentStateCheckpoint.save() angelegt, also liefert
            # try_restore() dauerhaft False -- der Patch faellt dann bei
            # JEDEM Rollback auf den urspruenglichen kv_cache_seq_rm
            # zurueck, der bei rekurrentem State strukturell fehlschlaegt.
            # Genau das fuehrte zuvor dazu, dass auch der Retry ohne
            # Speculative Decoding denselben Fehler erneut warf. Fix: den
            # Original-Callable wiederherstellen, sobald kein Draft-Modell
            # mehr aktiv ist.
            if self._original_kv_cache_seq_rm is not None:
                self.llm._ctx.kv_cache_seq_rm = self._original_kv_cache_seq_rm
                self._original_kv_cache_seq_rm = None
                self.log(
                    "Hinweis: kv_cache_seq_rm-Patch fuer Hybrid-/SSM-"
                    "Checkpointing zurueckgesetzt. Fuer dieses Modell "
                    "bedeutet das: Praefix-Wiederverwendung ueber "
                    "mehrere Turns hinweg ist ab jetzt eingeschraenkt "
                    "(rekurrenter State ist nicht partiell zuruecknehmbar) "
                    "-- bei Bedarf Modell neu laden, um Checkpoint-"
                    "basiertes Speculative Decoding wieder zu aktivieren."
                )
            return self.generate_stream(
                messages, max_new_tokens, temperature, top_p, callback,
                reasoning_callback=reasoning_callback, tools=tools,
                tool_choice=tool_choice, max_reasoning_seconds=max_reasoning_seconds,
                repeat_penalty=repeat_penalty, stop_event=stop_event,
                _retry_without_spec=True, _retry_hard_reset=hard_reset_retried,
            )

        # ZWEITER, unabhaengiger Fallback: der obige Draft-Modell-Retry greift
        # nur, wenn tatsaechlich noch ein Draft-Modell aktiv war. Dieser Fehler
        # tritt aber nachweislich AUCH auf, wenn Speculative Decoding laengst
        # deaktiviert ist (z.B. schon in einem frueheren Turn derselben
        # Sitzung abgeschaltet) -- dann ist die wahrscheinlichste Ursache ein
        # State-Desync zwischen llama-cpp-python's Python-Buchhaltung
        # (self.n_tokens/self.scores) und dem tatsaechlichen nativen KV-/
        # rekurrenten State (siehe _hard_reset_hybrid_context-Docstring).
        # Einmaliger harter Reset + Neuversuch, BEVOR endgueltig aufgegeben
        # wird.
        if not hard_reset_retried:
            self.log(
                "Speculative Decoding ist bereits deaktiviert, der Fehler "
                "tritt trotzdem auf -- das deutet auf einen State-Rollback-"
                "Bug hin, nicht auf das Draft-Modell. Versuche einen harten "
                "Kontext-Reset (echtes Leeren von KV-/rekurrentem State auf "
                "C-Ebene statt nur des Python-Zaehlers) und generiere die "
                "Anfrage einmal neu."
            )
            self._hard_reset_hybrid_context()
            return self.generate_stream(
                messages, max_new_tokens, temperature, top_p, callback,
                reasoning_callback=reasoning_callback, tools=tools,
                tool_choice=tool_choice, max_reasoning_seconds=max_reasoning_seconds,
                repeat_penalty=repeat_penalty, stop_event=stop_event,
                _retry_without_spec=True, _retry_hard_reset=True,
            )

        raise RuntimeError(
            "llama_decode ist auch nach Deaktivierung von Speculative "
            f"Decoding UND einem harten Kontext-Reset fehlgeschlagen ({exc}). "
            "Falls die Fehlermeldung 'could not broadcast input array ... "
            "into shape (0,)' lautet und dabei self._patch_draft_model_"
            "scores_ringbuffer NICHT beim Laden geloggt wurde ('WARNUNG: "
            "Scores-Ringpuffer-Patch konnte nicht angewendet werden'): dann "
            "ist der Scores-Puffer-Groessenbug (Llama.__init__ erzwingt "
            "effektives logits_all=True sobald draft_model gesetzt ist, "
            "dimensioniert den Puffer aber nur bei explizit gesetztem "
            "logits_all=True korrekt) NICHT gefixt, weil der Ringpuffer-"
            "Patch aus irgendeinem Grund fehlgeschlagen ist -- siehe die "
            "WARNUNG-Zeile beim Laden fuer den genauen Grund (z.B. eine "
            "llama-cpp-python-Version, deren eval()-Struktur nicht mehr zum "
            "gepatchten Code passt). Wahrscheinlichste verbleibende "
            "Ursachen sonst: (1) n_ctx (" + str(self.n_ctx) + ") zu gross "
            "fuer verfuegbaren VRAM/RAM bei n_gpu_layers>0; (2) "
            "kv_cache_type='q8_0' in Kombination mit den geladenen "
            "GPU-Layern nicht unterstuetzt; (3) bei echten Hybrid-/SSM-"
            "Modellen (Mamba/Qwen3.5-artig) ein State-Desync zwischen "
            "llama-cpp-python's Python-Buchhaltung und dem experimentellen "
            "nativen n_rs_seq-Rollback fuer rekurrenten State -- am "
            "sichersten auszuschliessen durch Neuladen des Modells OHNE "
            "native Hybrid-Rollback (try_enable_native_recurrent_rollback "
            "nicht aufrufen). Zum Eingrenzen: n_ctx deutlich senken (z.B. "
            "8192) und/oder kv_cache_type auf 'f16' stellen und/oder Modell "
            "ohne native Hybrid-Rollback laden."
        ) from exc

    def generate_stream(self, messages, max_new_tokens, temperature, top_p, callback,

                         reasoning_callback=None, tools=None, tool_choice=None,

                         max_reasoning_seconds=0, repeat_penalty=1.1, stop_event=None,

                         _retry_without_spec=False, _retry_hard_reset=False):

        """Streaming-Generierung. Tokens werden sofort per callback()/

        reasoning_callback() ausgegeben, waehrend das Modell noch generiert

        (nicht erst wenn alles fertig ist).



        max_reasoning_seconds (0 = unbegrenzt): sobald das Reasoning

        (<think>/reasoning_content) laenger als diese Zeitspanne laeuft,

        wird die Generierung sofort abgebrochen -- die 'Denkzeit-Begrenzung'

        nach Uhrzeit statt Token-Anzahl. Die Zeit laeuft ab dem ersten

        Reasoning-Token, nicht ab Beginn der ganzen Anfrage.



        Gibt (text, finish_reason, reasoning_limit_hit, tool_calls) zurueck.

        tool_calls ist nur gesetzt, falls das Chat-Format native Tool-Calls

        ueber den nicht gestreamten Fallback liefert (siehe unten).

        """

        if self.llm is None:

            raise RuntimeError("Modell ist noch nicht geladen.")

        if stop_event is not None and stop_event.is_set():
            # Schon vor dem Start gestoppt (z.B. Stop waehrend Tool-Runde
            # zwischen zwei Generierungsschritten gedrueckt).
            return "", "user_stop", False, None

        if self.is_hybrid_ssm:
            # Bei Mamba-/SSM-/Hybrid-Architekturen (z.B. Qwen3.5) ist der
            # interne State nicht wie ein normaler Transformer-KV-Cache
            # partiell zuruecknehmbar. llama-cpp-python's eingebautes
            # Live-Praefix-Matching (siehe _prepend_to_last_message) geht
            # aber genau davon aus, dass es bei einem Teil-Mismatch einfach
            # den abweichenden Teil "zuruecknehmen" und neu aufbauen kann.
            # Bei diesen Architekturen fuehrt das zuverlaessig zu
            # "inconsistent sequence positions" (X < Y verletzt) bzw.
            # "partial kv removal not supported". Der sichere Weg: den
            # Kontext VOR jedem Turn explizit zuruecksetzen, sodass
            # llama.cpp gar nicht erst versucht, ein (moeglicherweise nicht
            # mehr passendes) Praefix wiederzuverwenden, sondern den
            # kompletten uebergebenen messages-Verlauf sauber von vorne neu
            # durchrechnet. Kostet mehr Prefill-Zeit pro Turn (kein
            # Praefix-Caching mehr moeglich), ist bei dieser Architektur
            # aber die einzige zuverlaessige Option.
            self._hard_reset_hybrid_context()

        kwargs = dict(

            messages=messages,

            max_tokens=max_new_tokens,

            temperature=temperature,

            top_p=top_p,

            repeat_penalty=repeat_penalty,

            stream=True,

        )

        if tools:

            kwargs["tools"] = tools

            kwargs["tool_choice"] = tool_choice or "auto"



        try:

            stream = self.llm.create_chat_completion(**kwargs)

        except RuntimeError as e:

            # llama_decode returned -1 (o.ae.) -- kann direkt beim Aufruf

            # auftreten, wenn create_chat_completion nicht gestreamt/lazy

            # ist. Gleiche Behandlung wie unten im Streaming-Fall.

            return self._handle_decode_failure(

                e, messages, max_new_tokens, temperature, top_p, callback,

                reasoning_callback, tools, tool_choice, max_reasoning_seconds,

                repeat_penalty, stop_event, _retry_without_spec, _retry_hard_reset,

            )

        except ValueError:

            # Einige eingebaute Chat-Formate (z.B. "chatml-function-calling")

            # unterstuetzen kein Streaming, solange tool_choice="auto" ist.

            # Fallback: normale, nicht gestreamte Antwort -- dann erscheint

            # der Text zwar erst am Stueck, aber wenigstens bricht nichts ab.

            # Ein Zeitlimit kann hier nicht mitten in der Generierung

            # greifen (kein Stream), sondern nur ueber max_tokens vorbeugen.

            kwargs["stream"] = False

            response = self.llm.create_chat_completion(**kwargs)

            choice0 = response["choices"][0]

            msg = choice0["message"]

            raw_content = msg.get("content") or ""

            splitter = ThinkTagSplitter()

            answer_parts = []

            for is_reasoning, piece in splitter.feed(raw_content) + splitter.flush():

                if not piece:

                    continue

                if is_reasoning:

                    if reasoning_callback:

                        reasoning_callback(piece)

                else:

                    answer_parts.append(piece)

            content = "".join(answer_parts)

            if content:

                callback(content)

            return content, choice0.get("finish_reason"), False, msg.get("tool_calls")



        finish_reason = None

        full_text_parts = []

        reasoning_started_at = None

        reasoning_limit_hit = False

        splitter = ThinkTagSplitter()

        chunk_iter = iter(stream)

        while True:

            try:

                chunk = next(chunk_iter)

            except StopIteration:

                break

            except (RuntimeError, ValueError) as e:

                # llama_decode returned -1 mitten im Streaming (das ist der

                # Fall aus der urspruenglichen Fehlermeldung: der Generator

                # ist lazy, der Fehler kommt also erst hier beim Iterieren,

                # nicht schon beim create_chat_completion(**kwargs)-Aufruf

                # oben). Bereits gestreamter Text bleibt sichtbar; danach

                # wird -- einmalig -- ohne das experimentelle Speculative-

                # Decoding-Draft-Modell neu generiert, weil dessen

                # Kandidaten-Vorschlaege die wahrscheinlichste Fehlerquelle

                # sind.

                #

                # Neben RuntimeError faengt dies bewusst auch ValueError ab:

                # llama-cpp-python meldet einen Kontext-Ueberlauf durch das

                # Draft-Modell nicht immer als "llama_decode returned -1",

                # sondern manchmal als numpy-Broadcast-Fehler beim Schreiben

                # der Logits ("could not broadcast input array from shape

                # (N,) into shape (0,)"), weil der Ziel-Slice fuer

                # n_past+n_tokens > n_ctx bereits leer ist. Fachlich ist das

                # derselbe Fehlerfall wie der RuntimeError -- also identische

                # Behandlung (Speculative Decoding aus, einmal neu

                # generieren).

                if full_text_parts:

                    callback("\n\n[Generierung abgebrochen -- siehe Log.]")

                return self._handle_decode_failure(

                    e, messages, max_new_tokens, temperature, top_p, callback,

                    reasoning_callback, tools, tool_choice, max_reasoning_seconds,

                    repeat_penalty, stop_event, _retry_without_spec, _retry_hard_reset,

                )

            if stop_event is not None and stop_event.is_set():
                # Nutzer hat auf "Stop" gedrueckt -- Generierung sofort
                # abbrechen (z.B. wenn das Modell komplett abdriftet).
                finish_reason = "user_stop"
                break

            choice = chunk["choices"][0]

            delta = choice["delta"]

            finish_reason = choice.get("finish_reason") or finish_reason



            # Falls das Chat-Format reasoning_content doch mal separat

            # liefert (seltene eingebaute Formate), direkt uebernehmen.

            native_reasoning = delta.get("reasoning_content")

            if native_reasoning:

                if reasoning_started_at is None:

                    reasoning_started_at = time.monotonic()

                if reasoning_callback:

                    reasoning_callback(native_reasoning)

                if max_reasoning_seconds and (time.monotonic() - reasoning_started_at) >= max_reasoning_seconds:

                    reasoning_limit_hit = True

                    break



            token_text = delta.get("content")

            if token_text:

                for is_reasoning, piece in splitter.feed(token_text):

                    if not piece:

                        continue

                    if is_reasoning:

                        if reasoning_started_at is None:

                            reasoning_started_at = time.monotonic()

                        if reasoning_callback:

                            reasoning_callback(piece)

                    else:

                        full_text_parts.append(piece)

                        callback(piece)

                if (splitter.in_think and reasoning_started_at is not None

                        and max_reasoning_seconds

                        and (time.monotonic() - reasoning_started_at) >= max_reasoning_seconds):

                    reasoning_limit_hit = True

                    break



        if reasoning_limit_hit:

            minutes = max_reasoning_seconds / 60

            callback(

                f"\n\n[Denkzeit-Limit erreicht ({minutes:.1f} Minuten) -- "

                "Generierung wurde abgebrochen, bevor das Modell selbst "

                "fertig 'nachgedacht' hatte.]"

            )

        else:

            # Reststueck ausgeben, das noch im Splitter-Puffer haengt

            # (z.B. wenn der letzte Chunk mit einem angeschnittenen Tag endet).

            for is_reasoning, piece in splitter.flush():

                if not piece:

                    continue

                if is_reasoning:

                    if reasoning_callback:

                        reasoning_callback(piece)

                else:

                    full_text_parts.append(piece)

                    callback(piece)

            if finish_reason == "length":

                callback(

                    "\n\n[Abgebrochen: max_tokens erreicht, bevor die eigentliche "

                    "Antwort fertig war. Bei Reasoning-Modellen 'Max. neue Tokens' "

                    "deutlich hoeher setzen (z.B. 2048-4096), da das Modell erst "

                    "denkt und danach erst antwortet.]"

                )

            elif finish_reason == "user_stop":

                callback("\n\n[Vom Nutzer gestoppt.]")



        return "".join(full_text_parts), finish_reason, reasoning_limit_hit, None





    def run_with_tools(self, messages, tools, max_new_tokens, temperature, top_p,

                        callback, reasoning_callback=None, tool_log=None,

                        max_reasoning_seconds=0, max_iterations=None,

                        repeat_penalty=1.1, stop_event=None):

        """

        Agent-Schleife: solange das Modell Tool-Calls anfordert, werden diese

        lokal ausgefuehrt und als 'tool'-Nachrichten zurueckgegeben. JEDER

        Schritt (auch Zwischenschritte mit Tool-Aufrufen) wird jetzt

        gestreamt, sodass du den Text schon waehrend der Generierung siehst

        und nicht erst wenn alles fertig ist.



        max_reasoning_seconds (0 = unbegrenzt) begrenzt die Denkzeit pro

        Schritt: wird die Grenze erreicht, bricht der aktuelle Schritt sofort

        ab und das Modell wird mit einem zusaetzlichen Hinweis gezwungen,

        direkt (ohne weitere Tools) eine finale Antwort zu liefern -- so

        haengt die Generierung nicht mehr unbegrenzt lange fest.



        Benoetigt ein Chat-Template mit Function-Calling-Unterstuetzung

        (z.B. Qwen2.5/Qwen3-, Hermes- oder Llama-3.1-Tool-Format).

        """

        if self.llm is None:

            raise RuntimeError("Modell ist noch nicht geladen.")



        max_iterations = max_iterations or TOOL_LOOP_MAX_ITERATIONS



        if not tools:

            text, finish_reason, _, _ = self.generate_stream(

                messages, max_new_tokens, temperature, top_p, callback,

                reasoning_callback, max_reasoning_seconds=max_reasoning_seconds,

                repeat_penalty=repeat_penalty, stop_event=stop_event,

            )

            messages.append({"role": "assistant", "content": text})

            return text



        # WICHTIG: 'messages' wird bewusst NICHT kopiert, sondern in-place

        # erweitert (Tool-Calls + Tool-Ergebnisse), damit der Aufrufer (die

        # persistente Chat-Historie) diese Zwischenschritte fuer den naechsten

        # Chat-Turn ebenfalls kennt.

        working_messages = messages



        # Token-Cache fuers Kontext-Budget: 'working_messages' wird pro
        # Tool-Runde nur AM ENDE erweitert, bereits vorhandene Eintraege
        # aendern sich nie mehr. Vorher wurde bei JEDER der bis zu
        # TOOL_LOOP_MAX_ITERATIONS Runden die komplette Liste neu
        # tokenisiert (quadratischer Aufwand ueber die Tool-Schleife hinweg).
        # Jetzt: schon bekannte Nachrichten (per id() identifiziert, da
        # working_messages nur angehaengt, nie mutiert oder umsortiert wird)
        # bleiben im Cache, es wird nur der neue Rest tokenisiert.
        _budget_tok_cache = {}

        def context_budget_exceeded():

            # Grober Schutz gegen den Kontext-Ueberlauf, der wahrscheinlichste

            # Ursache fuer harte Abstuerze (WinError/native Exception) nach

            # sehr vielen Tool-Runden: working_messages waechst mit jeder

            # Runde (Tool-Calls + Tool-Ergebnisse) weiter, ohne dass das je

            # gegen n_ctx geprueft wurde. llama.cpp faengt ein Ueberschreiten

            # von n_ctx nicht sauber ab, sondern kann mit einer nativen

            # Exception abstuerzen -- das pruefen wir hier lieber VORHER.

            try:

                total = 0
                for m in working_messages:
                    key = id(m)
                    t = _budget_tok_cache.get(key)
                    if t is None:
                        t = self.tokenize_count(m.get("content") or "")
                        _budget_tok_cache[key] = t
                    total += t

            except Exception:

                return False, 0

            needed = total + max_new_tokens + CONTEXT_SAFETY_MARGIN_TOKENS

            return needed > self.n_ctx, total



        for iteration in range(max_iterations):

            if stop_event is not None and stop_event.is_set():
                # Zwischen zwei Tool-Runden gestoppt -- gar nicht erst
                # weiter generieren.
                callback("\n\n[Vom Nutzer gestoppt.]")
                return ""

            exceeded, prompt_tokens = context_budget_exceeded()

            if exceeded:

                if tool_log:

                    tool_log(

                        "[Hinweis] Kontextfenster reicht nicht: "

                        f"Prompt/Verlauf (~{prompt_tokens} Tokens) + "

                        f"'Max. neue Tokens' ({max_new_tokens}) + Sicherheitspuffer "

                        f"({CONTEXT_SAFETY_MARGIN_TOKENS}) > n_ctx ({self.n_ctx}). "

                        "'Max. neue Tokens' muss deutlich kleiner als n_ctx sein, "

                        "da sich Prompt und Antwort dasselbe Fenster teilen -- "

                        "breche Tool-Schleife ab, bevor es knallt."

                    )

                break



            text, finish_reason, reasoning_limit_hit, native_tool_calls = self.generate_stream(

                working_messages, max_new_tokens, temperature, top_p, callback,

                reasoning_callback, tools=tools, tool_choice="auto",

                max_reasoning_seconds=max_reasoning_seconds,

                repeat_penalty=repeat_penalty, stop_event=stop_event,

            )

            content = text

            tool_calls = native_tool_calls



            if finish_reason == "user_stop":
                # Sofort abbrechen -- anders als beim Denkzeit-Limit KEIN
                # erzwungener "Antworte jetzt sofort"-Zusatzschritt, der
                # Nutzer wollte ja gerade, dass nichts mehr generiert wird.
                working_messages.append({"role": "assistant", "content": content})
                return content

            if reasoning_limit_hit:

                # Denkzeit-Limit erreicht: Modell zwingen, JETZT sofort ohne

                # weitere Tools/weiteres Nachdenken zu antworten.

                working_messages.append({"role": "assistant", "content": content})

                working_messages.append({

                    "role": "user",

                    "content": (

                        "[System] Denkzeit-Limit erreicht. Antworte jetzt "

                        "sofort mit deiner finalen Antwort in wenigen Saetzen, "

                        "ohne weiter nachzudenken und ohne weitere Tool-Aufrufe."

                    ),

                })

                final_text, _, _, _ = self.generate_stream(

                    working_messages, max(256, max_new_tokens // 4), temperature,

                    top_p, callback, reasoning_callback,

                    repeat_penalty=repeat_penalty, stop_event=stop_event,

                )

                working_messages.append({"role": "assistant", "content": final_text})

                return final_text



            manual_calls = None

            if not tool_calls and ("<tool_call>" in content or "<tool_code>" in content):

                manual_calls = parse_manual_tool_calls(content)



            if not tool_calls and not manual_calls:

                # Kein weiterer Tool-Call -> Text wurde bereits gestreamt,

                # jetzt nur noch in die Historie schreiben.

                working_messages.append({"role": "assistant", "content": content})

                if finish_reason == "length":

                    callback(

                        "\n\n[Abgebrochen: max_tokens erreicht. 'Max. neue "

                        "Tokens' erhoehen.]"

                    )

                elif re.search(r"<tool_(call|code)>(?!.*</tool_\1>)", content, re.DOTALL):

                    # Oeffnendes Tag ohne passendes schliessendes Tag im Text

                    # -- das Modell hat einen Tool-Aufruf begonnen, dann aber

                    # abgebrochen (z.B. eigenes Stop-Token mitten im Block).

                    # Bisher endete die Runde hier kommentarlos; jetzt gibt es

                    # wenigstens einen sichtbaren Hinweis statt stiller Stille.

                    callback(

                        "\n\n[Hinweis] Antwort endet mit einem nicht "

                        "abgeschlossenen Tool-Aufruf-Tag -- das Modell hat "

                        "wahrscheinlich vorzeitig gestoppt, statt den Aufruf "

                        "zu vervollstaendigen. Kein Tool wurde ausgefuehrt."

                    )

                return content



            if tool_calls:

                # Natives tool_calls-Format (llama.cpp/OpenAI-Struktur).

                working_messages.append({

                    "role": "assistant",

                    "content": content,

                    "tool_calls": tool_calls,

                })



                for call in tool_calls:

                    fn = call.get("function", {})

                    name = fn.get("name", "")

                    if not name or not call.get("id"):
                        # Leerer/kaputter Tool-Call-Fragment (z.B. durch
                        # abgebrochenes Streaming). Nicht ausfuehren und
                        # nicht in die History schreiben -- ein Tool-Call
                        # ohne Namen/ID im Verlauf laesst llama-server beim
                        # naechsten Request mit HTTP 500 abbrechen.
                        if tool_log:
                            tool_log(f"[Hinweis] Leerer/ungueltiger Tool-Call uebersprungen: {call}")
                        continue

                    raw_args = fn.get("arguments", "{}")

                    try:

                        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

                    except json.JSONDecodeError:

                        arguments = {}



                    if tool_log:

                        tool_log(f"[Tool-Aufruf] {name}({arguments})")



                    result = execute_tool(name, arguments)



                    if tool_log:

                        preview = result if len(result) < 500 else result[:500] + " [...]"

                        tool_log(f"[Tool-Ergebnis] {name}: {preview}")



                    working_messages.append({

                        "role": "tool",

                        "tool_call_id": call.get("id", ""),

                        "name": name,

                        "content": result,

                    })

            else:

                # Fallback: Modell hat Tool-Calls als reinen Text ausgegeben

                # (kein natives Chat-Template-Function-Calling). Wir haengen

                # die rohe Antwort trotzdem in die Historie, fuehren die

                # erkannten Aufrufe aus und schicken das Ergebnis als

                # normale User-Nachricht zurueck, damit auch Modelle ohne

                # 'tool'-Rollen-Unterstuetzung im Template weitermachen

                # koennen.

                if tool_log:

                    tool_log("[Hinweis] Tool-Call im Textformat erkannt (kein natives tool_calls-Feld).")

                working_messages.append({"role": "assistant", "content": content})



                result_texts = []

                for call in manual_calls:

                    name = call["name"]

                    arguments = call["arguments"]



                    if tool_log:

                        tool_log(f"[Tool-Aufruf] {name}({arguments})")



                    result = execute_tool(name, arguments)



                    if tool_log:

                        preview = result if len(result) < 500 else result[:500] + " [...]"

                        tool_log(f"[Tool-Ergebnis] {name}: {preview}")



                    result_texts.append(f"Ergebnis von {name}:\n{result}")



                working_messages.append({

                    "role": "user",

                    "content": (

                        "[Tool-Ergebnisse]\n" + "\n\n".join(result_texts) +

                        "\n\nBitte antworte jetzt basierend auf diesen Ergebnissen."

                    ),

                })



        callback(

            "\n\n[Abgebrochen: maximale Anzahl an Tool-Aufruf-Runden "

            f"({max_iterations}) erreicht.]"

        )

        return ""





# ---------------------------------------------------------------------------

# MemoryFile: das context.md-Prinzip. Persistente, ueber Sitzungen hinweg

# gueltige Erinnerung als Markdown-Datei. Anders als ChatStore (voller

# Rohverlauf) enthaelt diese Datei nur VERDICHTETE Zusammenfassungen, die

# das Modell selbst periodisch schreibt -- dadurch bleibt sie klein genug,

# um bei JEDEM Turn komplett als System-Kontext mitgegeben zu werden, egal

# wie alt/lang die Sitzung schon ist. Das ist der eigentliche Hebel fuer

# "viel effektiven Kontext, wenig n_ctx/VRAM": statt den ganzen Rohverlauf

# ins Fenster zu quetschen, geht nur eine kleine komprimierte Essenz rein.

# Schreibzugriff ist append-only (nie ueberschreiben/neu komprimieren) --

# vermeidet Korruption bei parallelen/abgebrochenen Schreibvorgaengen.

# ---------------------------------------------------------------------------
class LlamaServerEngine:
    """Alternative zu LlamaCppEngine: startet llama-server.exe als eigenen
    Subprozess und spricht per HTTP (OpenAI-kompatibel, /v1/chat/completions
    mit stream=True, SSE) mit ihm, statt das Modell im selben Python-Prozess
    ueber die eingebettete Llama-Klasse zu laden.

    Warum das noetig ist: MTP (Multi-Token Prediction, llama.cpp-PR #22673)
    ist Stand jetzt nur ueber die C++-Binaries llama-server/llama-cli
    verdrahtet (--spec-type draft-mtp --spec-draft-n-max N). Die Python-
    Low-Level-Bindings von llama-cpp-python kennen diesen Parameter noch
    nicht -- der einzige Weg an den echten MTP-Speedup ist deshalb, die
    tatsaechliche llama-server-Binary zu benutzen.

    Bietet dieselbe Kern-Schnittstelle wie LlamaCppEngine (n_ctx,
    tokenize_count, generate_stream mit identischer Signatur/Rueckgabe,
    draft_model=None als Kompatibilitaets-Attribut), damit der Rest der
    GUI (Tool-Loop, Memory-Injection, Effort-Prompt, Stop-Button etc.)
    unveraendert weiterlaeuft, egal welches Engine-Objekt gerade aktiv ist.
    Bewusst NICHT nachgebaut: die ganzen llama-cpp-python-spezifischen
    Workarounds (M-RoPE-Check, Hybrid-SSM-Hard-Reset, Draft-Model-
    Ringpuffer-Patch, numpy-Broadcast-Decode-Fehler) -- die betreffen alle
    Eigenheiten der Python-Bindings bzw. des eingebetteten Kontexts. Der
    Server verwaltet Kontext/Praefix-Cache/Context-Shifting intern selbst,
    genau wie llama-cli es tut.
    """

    def __init__(self, log_fn=None):
        self.log = log_fn or (lambda msg: None)
        self.process = None
        self.base_url = None
        self.n_ctx = 4096
        self.draft_model = None       # Kompatibilitaets-Attribut fuer _on_stop() etc.
        self.llm = None               # Kompatibilitaets-Attribut fuer run_with_tools()
                                       # ("ist geladen?"-Check) -- True sobald bereit.
        self.is_hybrid_ssm = False    # nicht relevant, Server managt das selbst
        self.is_mrope = False

    def load(self, model_path, n_gpu_layers, n_ctx, n_threads, n_batch=512,
              flash_attn=True, kv_cache_type="f16", n_ubatch=None,
              n_threads_batch=None, mtp_enabled=True, mtp_draft_n_max=3,
              server_binary_path="llama-server.exe", host="127.0.0.1",
              port=8910, startup_timeout_sec=120,
              chat_template_override=None):
        """Startet llama-server.exe als Subprozess und wartet, bis /health
        antwortet. Ein evtl. bereits laufender vorheriger Server wird zuerst
        beendet (siehe shutdown())."""
        self.shutdown()

        if not os.path.isfile(server_binary_path):
            raise FileNotFoundError(
                f"llama-server-Binary nicht gefunden unter: {server_binary_path}\n"
                "Lade eine aktuelle llama.cpp-Release (b9200 oder neuer, fuer "
                "MTP-Support) von https://github.com/ggml-org/llama.cpp/releases "
                "und trage den Pfad zur llama-server.exe im Feld oben ein."
            )

        base_args = [
            server_binary_path,
            "--model", model_path,
            "--host", host,
            "--port", str(port),
            "--n-gpu-layers", str(n_gpu_layers),
            "--ctx-size", str(n_ctx),
            "--batch-size", str(n_batch),
            "--threads", str(n_threads if n_threads > 0 else 4),
            "--parallel", "1",
            # Auto-Fit-Logik (-fit) fest deaktiviert: sie versucht, unsere
            # explizit gesetzten n_gpu_layers/ctx-size selbst anzupassen,
            # gibt bei explizit gesetztem n_gpu_layers zwar nur eine
            # Warnung aus ("... already set by user to N, abort") -- diese
            # Kombination aus manuellem n_gpu_layers + --spec-type
            # draft-mtp bringt die Fit-Logik in diesem Build aber zum
            # Abstuerzen (bestaetigt reproduzierbar, unabhaengig von
            # --ctx-size -- siehe ggml-org/llama.cpp#23395). Wir brauchen
            # die Auto-Fit-Logik ohnehin nicht, da wir alle Werte schon
            # explizit vorgeben.
            "--fit", "off",
            # --jinja: aktiviert llama-server's Jinja2-Chat-Template-Engine
            # inkl. nativem OpenAI-Function-Calling-Parsing. OHNE diesen
            # Flag lehnt der Server 'tools'-Requests entweder direkt mit
            # "tools param requires --jinja flag" ab, oder das eingebettete
            # GGUF-Chat-Template wird gar nicht erst zum Rendern der
            # <tools>-Sektion im Prompt genutzt -- das Modell sieht die
            # Tool-Definitionen dann effektiv nie und antwortet (korrekt!),
            # es habe keinen Zugriff. Bei der alten In-Process-Engine
            # (llama-cpp-python) war das kein Thema, weil die dortige
            # create_chat_completion()-Funktion Function-Calling intern
            # anders (nicht ueber den Server-HTTP-Pfad) handhabt -- daher
            # trat der Unterschied erst seit dem Umstieg auf llama-server
            # auf.
            "--jinja",
        ]
        if n_ubatch:
            base_args += ["--ubatch-size", str(n_ubatch)]
        if n_threads_batch:
            base_args += ["--threads-batch", str(n_threads_batch)]
        if flash_attn:
            base_args += ["--flash-attn", "on"]
        if kv_cache_type != "f16":
            base_args += ["--cache-type-k", kv_cache_type, "--cache-type-v", kv_cache_type]
        # WICHTIG: --spec-type/--spec-draft-n-max NICHT hier in base_args
        # aufnehmen. Diese Flags werden weiter unten pro Kandidat (MTP an /
        # MTP aus) gezielt hinzugefuegt (mtp_args_variant) -- base_args ist
        # der gemeinsame Teil fuer BEIDE Varianten. Waeren sie schon hier
        # drin, wuerde der "MTP aus"-Fallback-Versuch MTP trotzdem mit
        # starten (siehe Bug: Server-Log zeigte "specified multiple times",
        # weil das Flag doppelt drin war, und der Fallback-Test war dadurch
        # nie wirklich MTP-frei).

        mtp_args_variant = ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(mtp_draft_n_max)]

        # candidates: (Label, MTP an?, Chat-Template-Override oder None)
        candidates = []
        mtp_states = [True, False] if mtp_enabled else [False]
        for use_mtp in mtp_states:
            label = "MTP an" if use_mtp else "MTP aus"
            candidates.append((label, use_mtp, chat_template_override))

        # Automatischer Diagnose-/Rettungs-Kandidat: NUR wenn der Aufrufer
        # nicht selbst schon einen Override vorgegeben hat, haengen wir
        # einen letzten Versuch an, der das im GGUF eingebettete Jinja-
        # Chat-Template durch das eingebaute "chatml"-Template ersetzt
        # (--chat-template chatml). Hintergrund (siehe Recherche oben):
        # bei gemergten Modellen kann tokenizer_config.json/Chat-Template
        # von einer anderen Komponente stammen als die eigentlichen
        # Token-Embeddings. Sieht das Modell dann die dazu nicht passende
        # Assistant-Rollen-Markierung aus dem (kaputten) eingebetteten
        # Template, haelt es diese faelschlich fuer ein Stop-Signal und
        # bricht nach 0-1 Tokens ab.
        if not chat_template_override:
            candidates.append(("MTP aus + Chat-Template-Override chatml", False, "chatml"))

        last_err = None
        last_probe_detail = None

        for label, use_mtp, tmpl_override in candidates:
            args = list(base_args)
            if use_mtp:
                args += mtp_args_variant
            if tmpl_override:
                # Ueberschreibt das eingebettete GGUF-Jinja-Template durch
                # ein von llama-server eingebautes Template (z.B. "chatml").
                args += ["--chat-template", tmpl_override]

            self.log(f"Starte llama-server ({label}): {' '.join(args)}")
            self._launch_process(args, None)
            self.base_url = f"http://{host}:{port}"
            self.n_ctx = n_ctx

            try:
                self._wait_until_ready(startup_timeout_sec)
            except TimeoutError as e:
                self.shutdown()
                last_err = e
                self.log(f"Start ({label}) nicht bereit geworden.")
                continue

            ok, detail = self._probe_generation_sane()
            if ok:
                if tmpl_override and tmpl_override != chat_template_override:
                    self.log(
                        "Diagnose bestaetigt: das im GGUF eingebettete Chat-"
                        "Template ist vermutlich das Problem (typisches Merge-"
                        "Artefakt -- Template und Token-Embeddings stammen aus "
                        "unterschiedlichen Ausgangsmodellen). Mit "
                        f"--chat-template {tmpl_override} generiert das Modell "
                        "wieder normal. MTP bleibt fuer diese Session "
                        "deaktiviert, da der Override MTP-Kompatibilitaet nicht "
                        "geprueft hat. Wenn das dauerhaft so bleiben soll, "
                        f"chat_template_override='{tmpl_override}' beim naechsten "
                        "Laden explizit mitgeben, um den Diagnose-Umweg zu "
                        "ueberspringen."
                    )
                elif not use_mtp and mtp_enabled:
                    self.log(
                        "MTP fuehrt bei diesem Modell zu sofortigem "
                        "Generierungsabbruch -- MTP fuer diese Session "
                        "automatisch deaktiviert."
                    )
                else:
                    self.log(f"llama-server bereit ({label}), Canary-Check ok: {detail}.")
                self.llm = True
                self._start_crash_watchdog()
                return

            last_probe_detail = detail
            self.log(f"Canary-Check fehlgeschlagen ({label}): {detail}.")
            self.shutdown()

        if last_probe_detail is None and last_err is not None:
            raise TimeoutError(
                f"llama-server wurde nach {startup_timeout_sec}s nicht bereit "
                f"(letzter Fehler: {last_err})."
            )
        detail_suffix = f" ({last_probe_detail})" if last_probe_detail else ""
        tried_template_override = any(tmpl for _, _, tmpl, _ in candidates)
        if tried_template_override:
            raise RuntimeError(
                "Generierung bricht sofort ab, unabhaengig von MTP an/aus, "
                f"der gesamten Vektor-Skalierungs-Leiter ({scale_ladder}){detail_suffix} "
                "UND einem Chat-Template-Override auf 'chatml'. Damit sind Control-"
                "Vector-Staerke, MTP und ein kaputtes eingebettetes Chat-Template als "
                "alleinige Ursache so gut wie ausgeschlossen. Wahrscheinlicher: "
                f"--ctx-size {n_ctx} liegt ausserhalb des fuer dieses Modell "
                "trainierten/gueltigen Bereichs, oder es liegt ein tieferliegender "
                "Merge-Schaden an den Token-Embeddings/Gewichten selbst vor (nicht "
                "nur am Template/Vektor). Teste das Modell probeweise mit deutlich "
                "kleinerem --ctx-size (z.B. 4096) und ganz ohne Vektor/MTP -- schlaegt "
                "es dann immer noch fehl, liegt es am Modell selbst."
            )
        raise RuntimeError(
            "Generierung bricht sofort ab, unabhaengig von MTP an/aus und "
            f"der gesamten Vektor-Skalierungs-Leiter ({scale_ladder}){detail_suffix}. Das liegt damit "
            "vermutlich nicht an MTP oder dem Vektor, sondern am Modell/"
            "Chat-Template selbst (z.B. Merge-Artefakt im gemergten Modell, "
            f"oder --ctx-size {n_ctx} sprengt den fuer dieses Modell gueltigen "
            "Bereich). Teste das Modell probeweise ohne Vektor UND ohne MTP "
            "-- schlaegt es dann immer noch fehl, liegt es am Modell selbst."
        )

    def _launch_process(self, args, cwd):
        """Startet llama-server.exe als Subprozess mit den gegebenen Args
        und pumpt dessen Stdout im Hintergrund ins Log. Wird sowohl fuer den
        finalen Start als auch fuer jeden Kalibrier-Versuch in load()
        verwendet."""
        try:
            self.process = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                cwd=cwd,
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                               if platform.system() == "Windows" else 0),
            )
        except OSError as e:
            raise RuntimeError(f"llama-server liess sich nicht starten: {e}")

        def _pump_output():
            if not self.process or not self.process.stdout:
                return
            for line in self.process.stdout:
                self.log("[llama-server] " + line.rstrip())
        threading.Thread(target=_pump_output, daemon=True).start()

    def _wait_until_ready(self, startup_timeout_sec):
        """Pollt /health, bis der gerade gestartete Prozess bereit ist.
        Wirft RuntimeError bei Absturz waehrend des Ladens bzw. TimeoutError
        wenn er innerhalb des Timeouts nicht bereit wird."""
        deadline = time.time() + startup_timeout_sec
        last_err = None
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server ist waehrend des Ladens abgestuerzt "
                    f"(Exit-Code {self.process.returncode}). Siehe Log oben."
                )
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=2) as resp:
                    if resp.status == 200:
                        return
            except Exception as e:
                last_err = e
            time.sleep(1.0)
        raise TimeoutError(str(last_err) if last_err else "kein /health-Response")

    def _probe_generation_sane(self, n_predict=24, timeout_sec=30):
        """Canary-Check nach dem Start: schickt einen kurzen Chat-Request an
        /v1/chat/completions (NICHT /completion) und prueft, ob tatsaechlich
        mehrere Tokens generiert werden.

        Wichtig: /completion nimmt einen rohen Prompt-String ohne
        Chat-Template. Bei --jinja erwartet das Modell aber Rollen-Markup
        aus dem GGUF-Template; ein ungetemplateter Rohstring fuehrt bei
        manchen Modellen zu einem einzelnen Interpunktions-Token gefolgt
        von sofortigem Stop -- unabhaengig von Control Vector oder MTP.
        Deshalb hier derselbe Pfad wie generate_stream() (Chat-Format)."""
        try:
            payload = json.dumps({
                "messages": [
                    {"role": "user", "content": "Beantworte kurz: Was ist die Hauptstadt von Frankreich?"}
                ],
                "max_tokens": n_predict,
                "temperature": 0.0,
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                self.base_url + "/v1/chat/completions", data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return False, f"Canary-Request fehlgeschlagen: {e}"

        choice = (data.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content") or "").strip()
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage") or {}
        n_generated = usage.get("completion_tokens")
        if n_generated is None:
            n_generated = len(content.split())
        if n_generated <= 1 or len(content) < 2:
            return False, (f"nur {n_generated} Token(s) generiert, "
                            f"finish_reason={finish_reason!r}, Content={content!r}")
        return True, f"{n_generated} Tokens generiert"

    def _start_crash_watchdog(self):
        """Ueberwacht den Server-Prozess NACH dem 'bereit'-Status. Ohne das
        hier wuerde ein spaeterer Absturz (z.B. CUDA-OOM beim Alloziieren
        der KV-Cache-/Compute-Puffer, oft ein harter Absturz ohne
        Fehlertext) komplett stillschweigend passieren -- der Log-Pump-
        Thread haette einfach aufgehoert, ohne dass irgendwo steht, WARUM."""
        proc = self.process

        def _watch():
            proc.wait()
            if self.process is not proc:
                return  # Prozess wurde regulaer per shutdown() ersetzt/beendet
            code = proc.returncode
            hint = ""
            # Haeufige Windows-Exit-Codes fuer harte Abstuerze uebersetzen.
            if code in (-1073741819, 3221225477):  # 0xC0000005
                hint = " (Access Violation -- typisch fuer CUDA-Out-of-Memory beim Alloziieren grosser Kontext-/KV-Cache-Puffer, z.B. bei sehr hohem --ctx-size)."
            elif code in (-1073740791, 3221226505):  # 0xC0000409 STATUS_STACK_BUFFER_OVERRUN
                hint = " (Stack-Buffer-Overrun-Schutz ausgeloest.)"
            self.log(
                f"FEHLER: llama-server ist unerwartet beendet worden "
                f"(Exit-Code {code}){hint}\n"
                "Haeufigste Ursache bei MTP + grossem --ctx-size: CUDA-VRAM "
                "reicht nicht fuer Haupt-KV-Cache + MTPs eigenen "
                "Zusatz-KV-Cache + Compute-Puffer. Abhilfe: --ctx-size stark "
                "reduzieren (z.B. testweise auf 8192) um zu pruefen, ob es "
                "grundsaetzlich laeuft, dann schrittweise hochsetzen."
            )
            self.base_url = None
            self.llm = None

        threading.Thread(target=_watch, daemon=True).start()

    def shutdown(self):
        """Beendet einen laufenden Server-Subprozess, falls vorhanden --
        wird vor jedem Neuladen und beim Programmende aufgerufen."""
        self.llm = None
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        except Exception:
            pass
        self.process = None

    def tokenize_count(self, text):
        """Fragt die Token-Anzahl beim Server ab (/tokenize). Bei Fehlern
        grobe Schaetzung (4 Zeichen/Token) als Fallback, damit die
        Fenster-Budget-Rechnung nicht hart abbricht."""
        try:
            payload = json.dumps({"content": text}).encode("utf-8")
            req = urllib.request.Request(
                self.base_url + "/tokenize", data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return len(data.get("tokens", []))
        except Exception:
            return max(1, len(text) // 4)

    def generate_stream(self, messages, max_new_tokens, temperature, top_p, callback,
                         reasoning_callback=None, tools=None, tool_choice=None,
                         max_reasoning_seconds=0, repeat_penalty=1.1, stop_event=None,
                         **_ignored):
        """Streamt ueber /v1/chat/completions (SSE) vom llama-server.
        Gleiche Rueckgabe wie LlamaCppEngine.generate_stream:
        (text, finish_reason, reasoning_limit_hit, tool_calls)."""
        if self.base_url is None:
            raise RuntimeError("Server ist noch nicht gestartet.")
        if stop_event is not None and stop_event.is_set():
            return "", "user_stop", False, None

        payload = {
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        full_text_parts = []
        finish_reason = None
        reasoning_started_at = None
        reasoning_limit_hit = False
        tool_calls = None
        splitter = ThinkTagSplitter()

        # WICHTIG: 'timeout' hier ist ein IDLE-Timeout pro Socket-Read (Zeit
        # zwischen zwei empfangenen Chunks), keine Gesamtzeit fuer die ganze
        # Antwort. 'max_new_tokens' (eine Tokenanzahl, typischerweise
        # 256-100000) direkt als Sekunden-Wert zu uebergeben war ein Bug:
        # bei kleinem max_new_tokens (z.B. 60-300) kann das Idle-Timeout
        # kuerzer sein als die Zeit, die das Modell bei langsamer
        # Prompt-Verarbeitung oder niedriger tg-Rate zwischen zwei Chunks
        # tatsaechlich braucht -- der Socket-Read wirft dann eine
        # (bisher ungefangene) Exception MITTEN im Stream, der Thread
        # bricht ab, und es sieht fuer den Nutzer wie ein zufaelliges,
        # kommentarloses Ende aus. Stattdessen: fester, grosszuegiger
        # Idle-Timeout, unabhaengig von max_new_tokens.
        STREAM_IDLE_TIMEOUT_SEC = 300

        try:
            resp = urllib.request.urlopen(req, timeout=STREAM_IDLE_TIMEOUT_SEC)
        except urllib.error.HTTPError as e:
            # HTTPError ist eine URLError-Unterklasse, muss also VOR dem
            # allgemeinen URLError-Except stehen. llama-server schreibt den
            # eigentlichen Grund (z.B. "tools param requires --jinja flag")
            # in den Response-Body, nicht in e selbst -- ohne den hier
            # auszulesen sah man bisher nur ein nichtssagendes "HTTP Error
            # 400: Bad Request".
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            raise RuntimeError(
                f"llama-server lehnte die Anfrage ab (HTTP {e.code}): {body or e.reason}"
            )
        except urllib.error.URLError as e:
            raise RuntimeError(f"Verbindung zu llama-server fehlgeschlagen: {e}")

        stream_error = None
        try:
            try:
                for raw_line in resp:
                    if stop_event is not None and stop_event.is_set():
                        finish_reason = "user_stop"
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choice0 = (chunk.get("choices") or [{}])[0]
                    delta = choice0.get("delta") or {}
                    piece_raw = delta.get("content") or ""
                    if delta.get("tool_calls"):
                        if tool_calls is None:
                            tool_calls = []
                        for tc_delta in delta["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            while len(tool_calls) <= idx:
                                tool_calls.append({
                                    "id": "", "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                })
                            slot = tool_calls[idx]
                            if tc_delta.get("id"):
                                slot["id"] = tc_delta["id"]
                            if tc_delta.get("type"):
                                slot["type"] = tc_delta["type"]
                            fn_delta = tc_delta.get("function") or {}
                            if fn_delta.get("name"):
                                slot["function"]["name"] += fn_delta["name"]
                            if fn_delta.get("arguments"):
                                slot["function"]["arguments"] += fn_delta["arguments"]
                    if choice0.get("finish_reason"):
                        finish_reason = choice0["finish_reason"]
                    if not piece_raw:
                        continue
                    for is_reasoning, piece in splitter.feed(piece_raw):
                        if not piece:
                            continue
                        if is_reasoning:
                            if reasoning_started_at is None:
                                reasoning_started_at = time.time()
                            if (max_reasoning_seconds and reasoning_started_at is not None
                                    and time.time() - reasoning_started_at > max_reasoning_seconds):
                                reasoning_limit_hit = True
                                finish_reason = "reasoning_time_limit"
                                break
                            if reasoning_callback:
                                reasoning_callback(piece)
                        else:
                            full_text_parts.append(piece)
                            callback(piece)
                    if reasoning_limit_hit:
                        break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                # Verbindung ist MITTEN im Stream abgerissen (Idle-Timeout,
                # Connection reset, Server abgestuerzt, etc.). Vorher
                # propagierte das ungefangen bis in den worker()-Thread und
                # der Turn endete kommentarlos, ohne dass der bereits
                # gestreamte Text als soweit fertig markiert wurde. Jetzt:
                # bereits empfangener Text bleibt erhalten, und es gibt
                # einen sichtbaren Hinweis statt stiller Stille.
                stream_error = e
                finish_reason = "stream_error"
        finally:
            try:
                resp.close()
            except Exception:
                pass

        if stream_error is not None:
            callback(
                f"\n\n[Hinweis] Verbindung zum llama-server ist waehrend des "
                f"Streamings abgebrochen ({stream_error}). Bereits generierter "
                f"Text oben ist vollstaendig, der Rest fehlt -- ggf. "
                f"'Fortsetzen' oder erneut generieren."
            )

        for is_reasoning, piece in splitter.flush():
            if piece and not is_reasoning:
                full_text_parts.append(piece)
                callback(piece)

        return "".join(full_text_parts), finish_reason, reasoning_limit_hit, tool_calls


# run_with_tools() ist reine Orchestrierungslogik (Tool-Aufruf-Schleife,
# Kontext-Budget-Check, manuelles Tool-Call-Parsing) und haengt nur von
# self.llm (nur fuer den "ist geladen?"-Check -- LlamaServerEngine setzt
# das kompatibel auf True/None, siehe load()/shutdown() oben),
# self.generate_stream, self.tokenize_count und self.n_ctx ab. Alle vier
# hat LlamaServerEngine bereits mit identischer Signatur -- die Methode
# hier zu duplizieren wuerde nur Divergenz-Risiko schaffen, deshalb wird
# sie 1:1 von LlamaCppEngine uebernommen.
LlamaServerEngine.run_with_tools = LlamaCppEngine.run_with_tools


# ---------------------------------------------------------------------------

class MemoryFile:

    def __init__(self, memory_dir, filename="context.md"):

        self.memory_dir = memory_dir

        os.makedirs(memory_dir, exist_ok=True)

        self.path = os.path.join(memory_dir, filename)



    def read_all(self):

        if not os.path.exists(self.path):

            return ""

        try:

            with open(self.path, "r", encoding="utf-8") as f:

                return f.read()

        except Exception:

            return ""



    def append_entry(self, summary_text):

        summary_text = (summary_text or "").strip()

        if not summary_text:

            return

        ts = time.strftime("%Y-%m-%d %H:%M")

        block = f"\n## Update {ts}\n{summary_text}\n"

        with open(self.path, "a", encoding="utf-8") as f:

            f.write(block)



    def size_chars(self):

        return len(self.read_all())





SUMMARIZE_PROMPT = (

    "Faelle die obige Konversation auf das Wichtigste zusammen, als "

    "kompakte Markdown-Stichpunkte fuer eine persistente Projekt-"

    "Gedaechtnisdatei. Nur neue/geaenderte Fakten seit dem letzten Stand, "

    "keine Wiederholung von bereits im Memory-Kontext oben Stehendem. "

    "Struktur:\n"

    "## Globale Entscheidungen\n- ...\n"

    "## Offene Aufgaben / Letzter Stand\n- [x] erledigt\n- [/] in Arbeit\n- [ ] offen\n"

    "Keine Erklaerungen, keine Einleitung, nur die Stichpunkte."

)





class App(tk.Tk):

    def __init__(self):

        super().__init__()

        self.title("Low-VRAM GGUF Inference GUI (llama-cpp-python)")

        self.geometry("940x880")



        self.engine = LlamaCppEngine(self.log)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.attachments = []  # Liste von Dateipfaden

        session_dir = os.path.join(DEFAULT_CHAT_SESSIONS_DIR, time.strftime("session_%Y%m%d_%H%M%S"))

        self.chat_store = ChatStore(session_dir)  # kompletter Verlauf lebt auf der Platte

        self.memory_file = MemoryFile(os.path.join(os.getcwd(), "ai_memory"))

        self.stop_event = threading.Event()  # per "Stop"-Button gesetzt, siehe _on_stop()

        self._build_ui()



    # ---------- UI ----------

    def _build_ui(self):

        pad = {"padx": 6, "pady": 4}



        # ---- Scrollbarer Container: alles unten haengt an "content" statt

        # an "self", damit die ganze GUI per Scrollbar/Mausrad durchscrollt,

        # egal wie viele Frames (Memory, Tools, etc.) noch dazukommen. ----

        outer = ttk.Frame(self)

        outer.pack(fill="both", expand=True)



        canvas = tk.Canvas(outer, highlightthickness=0)

        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)

        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side="left", fill="both", expand=True)

        vscroll.pack(side="right", fill="y")



        content = ttk.Frame(canvas)

        content_window = canvas.create_window((0, 0), window=content, anchor="nw")



        def _on_content_configure(event):

            canvas.configure(scrollregion=canvas.bbox("all"))

        content.bind("<Configure>", _on_content_configure)



        def _on_canvas_configure(event):

            # Innerer Frame soll immer die volle Canvas-Breite nutzen

            canvas.itemconfig(content_window, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)



        def _on_mousewheel(event):

            # Windows/macOS liefern event.delta, Linux X11 nutzt Button-4/5

            if event.num == 4:

                canvas.yview_scroll(-1, "units")

            elif event.num == 5:

                canvas.yview_scroll(1, "units")

            else:

                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")



        def _bind_mousewheel(_event):

            canvas.bind_all("<MouseWheel>", _on_mousewheel)

            canvas.bind_all("<Button-4>", _on_mousewheel)

            canvas.bind_all("<Button-5>", _on_mousewheel)



        def _unbind_mousewheel(_event):

            canvas.unbind_all("<MouseWheel>")

            canvas.unbind_all("<Button-4>")

            canvas.unbind_all("<Button-5>")



        # Mausrad nur aktiv, waehrend der Zeiger ueber der Canvas ist --

        # verhindert, dass Scrollen innerhalb der Text-Boxen mit dem

        # Seiten-Scroll kollidiert.

        canvas.bind("<Enter>", _bind_mousewheel)

        canvas.bind("<Leave>", _unbind_mousewheel)



        # Ab hier ist "content" (statt "self") der Parent aller Frames.

        frame_cfg = ttk.LabelFrame(content, text="Model Configuration")

        frame_cfg.pack(fill="x", **pad)



        # GGUF-Datei

        ttk.Label(frame_cfg, text="GGUF file:").grid(row=0, column=0, sticky="w", **pad)

        self.model_path_var = tk.StringVar(

            value="D:/hf_out/Qwythos-9B-Claude-Mythos-5-1M-Q5_K_M.gguf"

        )

        ttk.Entry(frame_cfg, textvariable=self.model_path_var, width=60).grid(

            row=0, column=1, **pad

        )

        ttk.Button(frame_cfg, text="Browse", command=self._browse_model).grid(

            row=0, column=2, **pad

        )



        # GPU Layers

        ttk.Label(frame_cfg, text="GPU layers (0 = CPU only, 999 = everything on GPU):").grid(

            row=1, column=0, sticky="w", **pad

        )

        self.gpu_layers_var = tk.StringVar(value="20")

        ttk.Entry(frame_cfg, textvariable=self.gpu_layers_var, width=10).grid(

            row=1, column=1, sticky="w", **pad

        )



        # Context size

        ttk.Label(frame_cfg, text="Context size (n_ctx):").grid(

            row=2, column=0, sticky="w", **pad

        )

        self.ctx_var = tk.StringVar(value="4096")

        ttk.Entry(frame_cfg, textvariable=self.ctx_var, width=10).grid(

            row=2, column=1, sticky="w", **pad

        )



        # Threads

        ttk.Label(frame_cfg, text="CPU threads (0 = automatic):").grid(

            row=3, column=0, sticky="w", **pad

        )

        self.threads_var = tk.StringVar(value="0")

        ttk.Entry(frame_cfg, textvariable=self.threads_var, width=10).grid(

            row=3, column=1, sticky="w", **pad

        )

        # n_threads_batch: SEPARATE Thread-Zahl fuer die Prefill-Phase
        # (Prompt-Processing), unabhaengig von n_threads (Decode-Phase).
        # llama.cpp behandelt beide Phasen intern unterschiedlich: Prefill
        # verarbeitet Tokens in Batches (grosse Matmuls) und skaliert bis zu
        # deutlich hoeheren Thread-Zahlen fast linear mit; Decode generiert
        # strikt ein Token nach dem anderen und ist dabei memory-bandwidth-
        # bound, nicht compute-bound -- zusaetzliche Threads bringen dort ab
        # der physischen Kernzahl kaum noch etwas und koennen durch
        # Synchronisations-Overhead sogar leicht bremsen. Bisher wurde
        # n_threads_batch nirgends gesetzt (blieb None -> llama.cpp nutzt
        # dafuer intern denselben Wert wie n_threads), wodurch Prefill
        # unnoetig langsam blieb, wenn n_threads bewusst niedrig gehalten
        # wurde. Leer = wie "CPU-Threads" oben (altes Verhalten).
        ttk.Label(frame_cfg, text="Threads for prefill (n_threads_batch, empty = same as above):").grid(
            row=3, column=2, sticky="w", **pad
        )
        self.threads_batch_var = tk.StringVar(value="")
        ttk.Entry(frame_cfg, textvariable=self.threads_batch_var, width=10).grid(
            row=3, column=3, sticky="w", **pad
        )



        # Batch size (beeinflusst die Groesse des Compute-Buffers, der bei

        # 'out of memory'-Fehlern beim Laden typischerweise zu groß ist)

        ttk.Label(frame_cfg, text="Batch size (n_batch, smaller = less VRAM):").grid(

            row=4, column=0, sticky="w", **pad

        )

        self.batch_var = tk.StringVar(value="512")

        ttk.Entry(frame_cfg, textvariable=self.batch_var, width=10).grid(

            row=4, column=1, sticky="w", **pad

        )



        # KV-Cache-Offload: der Haupthebel gegen genau den Fehler

        # 'ggml_backend_cuda_buffer_type_alloc_buffer ... out of memory' beim

        # Reservieren der Compute-/pp-Buffer. Deaktiviert haelt den

        # KV-Cache im normalen RAM statt im VRAM.

        self.offload_kqv_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(

            frame_cfg,

            text="Keep KV cache on GPU (disabling saves VRAM but is slower)",

            variable=self.offload_kqv_var,

        ).grid(row=5, column=0, columnspan=2, sticky="w", **pad)



        self.flash_attn_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(

            frame_cfg,

            text="Flash Attention (significantly reduces compute buffer size, if supported by the build)",

            variable=self.flash_attn_var,

        ).grid(row=6, column=0, columnspan=2, sticky="w", **pad)



        ttk.Label(frame_cfg, text="KV cache precision (smaller = smaller VRAM/RAM buffer):").grid(

            row=7, column=0, sticky="w", **pad

        )

        self.kv_cache_type_var = tk.StringVar(value="f16")

        ttk.Combobox(

            frame_cfg, textvariable=self.kv_cache_type_var,

            values=["f16", "q8_0", "q6_k", "q5_1", "q5_0", "q4_1", "q4_0"], state="readonly", width=8,

        ).grid(row=7, column=1, sticky="w", **pad)

        # n_ubatch: physische Prefill-Batchgroesse, getrennt von n_batch.
        # Leer/0 = wie n_batch. Kleiner reduziert VRAM-Spitzen beim
        # Prompt-Processing (z.B. nach Chunk-Eviction, siehe ChatStore).
        ttk.Label(frame_cfg, text="n_ubatch (prefill batch, empty = same as n_batch):").grid(
            row=8, column=0, sticky="w", **pad
        )
        self.ubatch_var = tk.StringVar(value="")
        ttk.Entry(frame_cfg, textvariable=self.ubatch_var, width=10).grid(
            row=8, column=1, sticky="w", **pad
        )

        self.speculative_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame_cfg,
            text="Speculative decoding (suffix index + continuation cache + token recycling, no extra VRAM)",
            variable=self.speculative_var,
        ).grid(row=9, column=0, columnspan=2, sticky="w", **pad)

        # HINWEIS: Der fruehere "Persistenter Prefix-Cache auf Platte"-
        # Schalter (LlamaDiskCache via set_cache) wurde entfernt -- er
        # verursachte einen M-RoPE-Positionsfehler im Zusammenspiel mit
        # Speculative Decoding (siehe Kommentar in Engine.load). Die
        # In-Prozess-Praefix-Wiederverwendung von llama-cpp-python bleibt
        # weiterhin automatisch aktiv, ohne Zusatzschalter.

        # MTP: nur sinnvoll, wenn das geladene GGUF selbst MTP-Zusatzkoepfe
        # enthaelt (z.B. "-MTP" im Dateinamen, siehe Engine.load-Kommentar
        # zu MTP fuer den aktuellen Stand -- Python-Bindings kennen den
        # Parameter Stand jetzt evtl. noch nicht, greift dann automatisch
        # nicht, ohne zu crashen).
        self.mtp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame_cfg,
            text="Use MTP (only if the GGUF has its own MTP heads -- only takes effect if llama-cpp-python supports it)",
            variable=self.mtp_var,
        ).grid(row=10, column=0, columnspan=2, sticky="w", **pad)

        ttk.Label(frame_cfg, text="MTP draft-n-max (number of proposed tokens per step):").grid(
            row=11, column=0, sticky="w", **pad
        )
        self.mtp_draft_n_max_var = tk.StringVar(value="3")
        ttk.Entry(frame_cfg, textvariable=self.mtp_draft_n_max_var, width=6).grid(
            row=11, column=1, sticky="w", **pad
        )

        # Server-Modus: startet llama-server.exe als Subprozess und spricht
        # per HTTP mit ihm, statt das GGUF im selben Python-Prozess ueber
        # die eingebettete Llama-Klasse zu laden (siehe LlamaServerEngine-
        # Docstring). Noetig fuer echtes MTP, da die Python-Bindings den
        # spec_type/mtp-Parameter noch nicht kennen -- llama-server schon.
        self.server_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame_cfg,
            text="Server mode (llama-server.exe as a subprocess -- needed for real MTP, recommended for MTP GGUFs)",
            variable=self.server_mode_var,
        ).grid(row=15, column=0, columnspan=2, sticky="w", **pad)

        ttk.Label(frame_cfg, text="Path to llama-server.exe:").grid(
            row=16, column=0, sticky="w", **pad
        )
        self.server_binary_var = tk.StringVar(value=r"D:\llama-b10258-bin-win-cuda-12.4-x64\llama-server.exe")
        ttk.Entry(frame_cfg, textvariable=self.server_binary_var, width=40).grid(
            row=16, column=1, sticky="w", **pad
        )
        ttk.Button(
            frame_cfg, text="Browse...",
            command=lambda: self.server_binary_var.set(
                filedialog.askopenfilename(
                    title="Select llama-server.exe",
                    filetypes=[("Ausfuehrbare Datei", "*.exe"), ("Alle Dateien", "*.*")],
                ) or self.server_binary_var.get()
            ),
        ).grid(row=16, column=2, sticky="w", **pad)

        ttk.Label(frame_cfg, text="Server port:").grid(
            row=17, column=0, sticky="w", **pad
        )
        self.server_port_var = tk.StringVar(value="8910")
        ttk.Entry(frame_cfg, textvariable=self.server_port_var, width=8).grid(
            row=17, column=1, sticky="w", **pad
        )

        self.load_btn = ttk.Button(frame_cfg, text="Load model", command=self._on_load)
        self.load_btn.grid(row=18, column=1, sticky="w", **pad)



        # ---- Dateianhaenge ----

        frame_files = ttk.LabelFrame(content, text="Attach files (content is sent along as context)")

        frame_files.pack(fill="x", **pad)



        btn_row = ttk.Frame(frame_files)

        btn_row.pack(fill="x", **pad)

        ttk.Button(btn_row, text="Add file", command=self._add_attachment).pack(side="left", padx=4)

        ttk.Button(btn_row, text="Remove selection", command=self._remove_attachment).pack(side="left", padx=4)

        ttk.Button(btn_row, text="Remove all", command=self._clear_attachments).pack(side="left", padx=4)



        self.attachment_list = tk.Listbox(frame_files, height=4)

        self.attachment_list.pack(fill="x", **pad)



        # ---- Tool-Zugriff ----

        frame_tools = ttk.LabelFrame(

            content, text="Tool access (requires a model/chat template with function-calling support)"

        )

        frame_tools.pack(fill="x", **pad)



        self.tools_enabled_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(

            frame_tools, text="Enable tools", variable=self.tools_enabled_var

        ).grid(row=0, column=0, sticky="w", **pad)



        self.tool_read_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(frame_tools, text="read_file", variable=self.tool_read_var).grid(

            row=0, column=1, sticky="w", **pad

        )

        self.tool_list_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(frame_tools, text="list_dir", variable=self.tool_list_var).grid(

            row=0, column=2, sticky="w", **pad

        )

        self.tool_write_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(frame_tools, text="write_file", variable=self.tool_write_var).grid(

            row=0, column=3, sticky="w", **pad

        )

        self.tool_python_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(

            frame_tools, text="run_python (caution: executes code!)",

            variable=self.tool_python_var,

        ).grid(row=1, column=0, columnspan=3, sticky="w", **pad)



        self.tool_diff_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(frame_tools, text="diff_files", variable=self.tool_diff_var).grid(

            row=1, column=3, sticky="w", **pad

        )



        self.tool_browser_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(

            frame_tools,

            text="Browser (Selenium, free navigation -- caution: controls a real Chrome browser!)",

            variable=self.tool_browser_var,

        ).grid(row=2, column=0, columnspan=4, sticky="w", **pad)



        # ---- Weitere Tools (allgemein, nicht browserbezogen) ----

        frame_tools2 = ttk.LabelFrame(content, text="Other tools (general)")

        frame_tools2.pack(fill="x", **pad)



        self.tool_http_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(frame_tools2, text="http_get (fetch a URL without a browser)", variable=self.tool_http_var).grid(

            row=0, column=0, sticky="w", **pad

        )

        self.tool_search_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(frame_tools2, text="web_search", variable=self.tool_search_var).grid(

            row=0, column=1, sticky="w", **pad

        )

        ttk.Label(frame_tools2, text="OpenSERP URL (empty = DuckDuckGo fallback):").grid(

            row=0, column=5, sticky="w", **pad

        )

        self.openserp_url_var = tk.StringVar(value="")

        ttk.Entry(frame_tools2, textvariable=self.openserp_url_var, width=26).grid(

            row=0, column=6, sticky="w", **pad

        )

        self.tool_calculate_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(frame_tools2, text="calculate", variable=self.tool_calculate_var).grid(

            row=0, column=2, sticky="w", **pad

        )

        self.tool_datetime_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(frame_tools2, text="get_datetime", variable=self.tool_datetime_var).grid(

            row=0, column=3, sticky="w", **pad

        )

        self.tool_sysinfo_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(frame_tools2, text="system_info", variable=self.tool_sysinfo_var).grid(

            row=0, column=4, sticky="w", **pad

        )

        self.tool_search_files_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(frame_tools2, text="search_files (grep)", variable=self.tool_search_files_var).grid(

            row=1, column=0, sticky="w", **pad

        )

        ttk.Label(

            frame_tools2,

            text=(

                "Tipp: OpenSERP lokal starten mit "

                "'docker run --rm -p 127.0.0.1:7000:7000 karust/openserp:latest "

                "serve -a 0.0.0.0 -p 7000', dann oben http://127.0.0.1:7000 eintragen."

            ),

            foreground="gray40",

        ).grid(row=2, column=0, columnspan=7, sticky="w", **pad)

        self.tool_append_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(frame_tools2, text="append_file", variable=self.tool_append_var).grid(

            row=1, column=1, sticky="w", **pad

        )

        self.tool_move_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(frame_tools2, text="move_file", variable=self.tool_move_var).grid(

            row=1, column=2, sticky="w", **pad

        )

        self.tool_delete_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(

            frame_tools2, text="delete_file (caution: deletes irrevocably!)",

            variable=self.tool_delete_var,

        ).grid(row=1, column=3, columnspan=2, sticky="w", **pad)



        # Generation-Parameter

        # ---- Chat-Verlauf (auf Platte, nicht im RAM) ----

        frame_hist = ttk.LabelFrame(

            content, text="Chat history (stored entirely on disk, not in RAM -- 'unlimited' context)"

        )

        frame_hist.pack(fill="x", **pad)



        ttk.Label(frame_hist, text="Session folder:").grid(row=0, column=0, sticky="w", **pad)

        self.session_dir_var = tk.StringVar(value=self.chat_store.session_dir)

        ttk.Entry(frame_hist, textvariable=self.session_dir_var, width=60, state="readonly").grid(

            row=0, column=1, columnspan=2, **pad

        )



        hist_btns = ttk.Frame(frame_hist)

        hist_btns.grid(row=1, column=0, columnspan=3, sticky="w", **pad)

        ttk.Button(hist_btns, text="New chat (new session)", command=self._on_new_chat).pack(side="left", padx=4)

        ttk.Button(hist_btns, text="Archive losslessly (.tar.xz)", command=self._on_archive_chat).pack(side="left", padx=4)

        ttk.Button(hist_btns, text="Load archive (.tar.xz)", command=self._on_load_archive).pack(side="left", padx=4)



        self.history_status_label = ttk.Label(

            frame_hist, text="Stored messages: 0  |  Chat turns: 0  |  In window sent to model: -"

        )

        self.history_status_label.grid(row=2, column=0, columnspan=3, sticky="w", **pad)



        # ---- Memory-File (context.md, sitzungsuebergreifend) ----

        frame_mem = ttk.LabelFrame(

            content, text="Memory file (ai_memory/context.md -- compressed, cross-session memory)"

        )

        frame_mem.pack(fill="x", **pad)



        self.memory_enabled_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(

            frame_mem, text="Include context.md as context on every turn + update periodically",

            variable=self.memory_enabled_var,

        ).grid(row=0, column=0, columnspan=2, sticky="w", **pad)



        ttk.Label(frame_mem, text="Auto-update every N user turns:").grid(row=1, column=0, sticky="w", **pad)

        self.memory_interval_var = tk.StringVar(value="6")

        ttk.Entry(frame_mem, textvariable=self.memory_interval_var, width=6).grid(

            row=1, column=1, sticky="w", **pad

        )



        mem_btns = ttk.Frame(frame_mem)

        mem_btns.grid(row=2, column=0, columnspan=3, sticky="w", **pad)

        ttk.Button(mem_btns, text="Show context.md now", command=self._on_show_memory).pack(side="left", padx=4)

        ttk.Button(mem_btns, text="Update context.md manually now", command=self._on_manual_summarize).pack(side="left", padx=4)



        self.memory_status_label = ttk.Label(frame_mem, text=self._memory_status_text())

        self.memory_status_label.grid(row=3, column=0, columnspan=3, sticky="w", **pad)



        frame_gen = ttk.LabelFrame(content, text="Generation")

        frame_gen.pack(fill="x", **pad)



        ttk.Label(frame_gen, text="Prompt:").grid(row=0, column=0, sticky="nw", **pad)

        self.prompt_box = scrolledtext.ScrolledText(frame_gen, width=90, height=5)

        self.prompt_box.grid(row=0, column=1, columnspan=3, **pad)

        self.prompt_box.insert("1.0", "Erklaere kurz, was lineare Attention ist.")



        ttk.Label(frame_gen, text="Max new tokens:").grid(row=1, column=0, sticky="w", **pad)

        self.max_tokens_var = tk.StringVar(value="2048")

        ttk.Entry(frame_gen, textvariable=self.max_tokens_var, width=10).grid(

            row=1, column=1, sticky="w", **pad

        )



        ttk.Label(frame_gen, text="Temperature:").grid(row=1, column=2, sticky="w", **pad)

        self.temp_var = tk.StringVar(value="0.7")

        ttk.Entry(frame_gen, textvariable=self.temp_var, width=10).grid(

            row=1, column=3, sticky="w", **pad

        )



        ttk.Label(frame_gen, text="Top-p:").grid(row=2, column=0, sticky="w", **pad)

        self.top_p_var = tk.StringVar(value="0.9")

        ttk.Entry(frame_gen, textvariable=self.top_p_var, width=10).grid(

            row=2, column=1, sticky="w", **pad

        )



        ttk.Label(frame_gen, text="Repeat penalty:").grid(row=2, column=2, sticky="w", **pad)

        self.repeat_penalty_var = tk.StringVar(value="1.1")

        ttk.Entry(frame_gen, textvariable=self.repeat_penalty_var, width=10).grid(

            row=2, column=3, sticky="w", **pad

        )



        ttk.Label(frame_gen, text="Max thinking time (minutes, 0=unlimited):").grid(

            row=3, column=0, sticky="w", **pad

        )

        self.max_reasoning_var = tk.StringVar(value="15")

        ttk.Entry(frame_gen, textvariable=self.max_reasoning_var, width=10).grid(

            row=3, column=1, sticky="w", **pad

        )



        ttk.Label(frame_gen, text="Max tool rounds:").grid(

            row=3, column=2, sticky="w", **pad

        )

        self.tool_rounds_var = tk.StringVar(value=str(TOOL_LOOP_MAX_ITERATIONS))

        ttk.Entry(frame_gen, textvariable=self.tool_rounds_var, width=10).grid(

            row=3, column=3, sticky="w", **pad

        )



        # Effort (analog zu Claudes "effort"-Parameter): steuert Denkzeit
        # und eine kurze System-Anweisung ans Modell, wie gruendlich es sein
        # soll. Aendert beim Auswaehlen automatisch "Max. Denkzeit" mit --
        # der Wert kann danach immer noch manuell ueberschrieben werden.
        ttk.Label(frame_gen, text="Effort:").grid(row=4, column=0, sticky="w", **pad)

        self.effort_var = tk.StringVar(value=DEFAULT_EFFORT)

        effort_combo = ttk.Combobox(
            frame_gen, textvariable=self.effort_var,
            values=list(EFFORT_PRESETS.keys()), state="readonly", width=8,
        )

        effort_combo.grid(row=4, column=1, sticky="w", **pad)

        effort_combo.bind("<<ComboboxSelected>>", self._on_effort_change)

        self.gen_btn = ttk.Button(

            frame_gen, text="Generate", command=self._on_generate, state="disabled"

        )

        self.gen_btn.grid(row=4, column=3, sticky="e", **pad)

        # Stop-Button: bricht eine laufende Generierung sofort ab (z.B. wenn
        # das Modell komplett abdriftet oder Unsinn labert). Startet
        # deaktiviert, wird von _on_generate() nur waehrend einer laufenden
        # Generierung aktiviert.
        self.stop_btn = ttk.Button(

            frame_gen, text="Stop", command=self._on_stop, state="disabled"

        )

        self.stop_btn.grid(row=4, column=2, sticky="w", **pad)



        # Output

        frame_out = ttk.LabelFrame(content, text="Output / Log")

        frame_out.pack(fill="both", expand=True, **pad)



        self.output_box = scrolledtext.ScrolledText(frame_out, wrap="word", height=22)

        self.output_box.pack(fill="both", expand=True, **pad)



    # ---------- Helpers ----------

    def _browse_model(self):

        path = filedialog.askopenfilename(

            title="Select GGUF file",

            filetypes=[("GGUF Dateien", "*.gguf"), ("Alle Dateien", "*.*")],

        )

        if path:

            self.model_path_var.set(path)

    def _add_attachment(self):

        paths = filedialog.askopenfilenames(title="Attach files")

        for p in paths:

            if p not in self.attachments:

                self.attachments.append(p)

                self.attachment_list.insert("end", p)



    def _remove_attachment(self):

        sel = list(self.attachment_list.curselection())

        for idx in reversed(sel):

            del self.attachments[idx]

            self.attachment_list.delete(idx)



    def _clear_attachments(self):

        self.attachments.clear()

        self.attachment_list.delete(0, "end")



    def log(self, msg):

        def append():

            self.output_box.insert("end", msg + "\n")

            self.output_box.see("end")



        self.after(0, append)



    # ---------- Actions ----------

    def _on_close(self):
        """Beim Fenster-Schliessen: laufenden llama-server-Subprozess (falls
        Server-Modus aktiv war) sauber beenden, sonst bleibt er als Orphan-
        Prozess haengen und blockiert den Port beim naechsten Start."""
        if isinstance(self.engine, LlamaServerEngine):
            self.engine.shutdown()
        self.destroy()

    def _on_load(self):

        self.load_btn.config(state="disabled")



        def worker():

            try:

                ubatch_raw = self.ubatch_var.get().strip()
                threads_batch_raw = self.threads_batch_var.get().strip()

                if self.server_mode_var.get():
                    # Server-Modus: alte Engine (egal welcher Typ) sauber
                    # beenden, falls es zuvor schon ein laufender Server war.
                    if isinstance(self.engine, LlamaServerEngine):
                        self.engine.shutdown()
                    self.engine = LlamaServerEngine(self.log)
                    self.engine.load(
                        model_path=self.model_path_var.get().strip(),
                        n_gpu_layers=int(self.gpu_layers_var.get()),
                        n_ctx=int(self.ctx_var.get()),
                        n_threads=int(self.threads_var.get()),
                        n_batch=int(self.batch_var.get()),
                        flash_attn=self.flash_attn_var.get(),
                        kv_cache_type=self.kv_cache_type_var.get(),
                        n_ubatch=int(ubatch_raw) if ubatch_raw else None,
                        n_threads_batch=int(threads_batch_raw) if threads_batch_raw else None,
                        mtp_enabled=self.mtp_var.get(),
                        mtp_draft_n_max=int(self.mtp_draft_n_max_var.get().strip() or "3"),
                        server_binary_path=self.server_binary_var.get().strip(),
                        port=int(self.server_port_var.get().strip() or "8910"),
                    )
                else:
                    if isinstance(self.engine, LlamaServerEngine):
                        self.engine.shutdown()
                        self.engine = LlamaCppEngine(self.log)
                    self.engine.load(
                        model_path=self.model_path_var.get().strip(),
                        n_gpu_layers=int(self.gpu_layers_var.get()),
                        n_ctx=int(self.ctx_var.get()),
                        n_threads=int(self.threads_var.get()),
                        n_batch=int(self.batch_var.get()),
                        offload_kqv=self.offload_kqv_var.get(),
                        flash_attn=self.flash_attn_var.get(),
                        kv_cache_type=self.kv_cache_type_var.get(),
                        n_ubatch=int(ubatch_raw) if ubatch_raw else None,
                        n_threads_batch=int(threads_batch_raw) if threads_batch_raw else None,
                        speculative_decoding=self.speculative_var.get(),
                        mtp_enabled=self.mtp_var.get(),
                        mtp_draft_n_max=int(self.mtp_draft_n_max_var.get().strip() or "3"),
                    )

                self.after(0, lambda: self.gen_btn.config(state="normal"))

            except ImportError:

                self.log(

                    "FEHLER: llama_cpp ist nicht installiert.\n"

                    "Installiere es mit: pip install llama-cpp-python\n"

                    "(Fuer GPU-Unterstuetzung brauchst du das CUDA-Build, siehe "

                    "https://github.com/abetlen/llama-cpp-python#installation "

                    "-> 'Installation with Hardware Acceleration'.)"

                )

                self.after(

                    0,

                    lambda: messagebox.showerror(

                        "Fehlendes Paket", "llama-cpp-python ist nicht installiert. Siehe Log."

                    ),

                )

            except Exception:

                err = traceback.format_exc()

                self.log("FEHLER beim Laden:\n" + err)

                if "out of memory" in err.lower() or "cudamalloc failed" in err.lower():

                    self.log(

                        "Hinweis: Das ist ein VRAM-Engpass beim Compute- bzw. KV-Cache-Puffer, "

                        "nicht bei den Modellgewichten. Abhilfe (eins nach dem anderen probieren):\n"

                        "  1. 'Kontextgroesse (n_ctx)' verkleinern (z.B. 2048 statt 4096) -- "

                        "der KV-Cache skaliert direkt damit\n"

                        "  2. 'KV-Cache-Praezision' auf q8_0 oder q4_0 stellen (halbiert/viertelt "

                        "genau den Puffer aus der Fehlermeldung)\n"

                        "  3. 'GPU-Layer' reduzieren (weniger Layer auf die GPU)\n"

                        "  4. 'KV-Cache auf GPU halten' deaktivieren\n"

                        "  5. 'Batch-Groesse (n_batch)' verkleinern (z.B. 256 oder 128)\n"

                        "Falls selbst n_gpu_layers=0 noch einen CUDA-Fehler wirft: dann liegt es "

                        "an der Hybrid-Attention-Architektur dieses Modells, die manche "

                        "Zustandspuffer ggf. immer auf die GPU legt -- das ist dann keine "

                        "Konfigurationsfrage mehr, sondern eine Limitierung des llama.cpp-Builds."

                    )

                self.after(

                    0,

                    lambda: messagebox.showerror(

                        "Ladefehler", "Modell konnte nicht geladen werden. Siehe Log."

                    ),

                )

            finally:

                self.after(0, lambda: self.load_btn.config(state="normal"))



        threading.Thread(target=worker, daemon=True).start()



    def _on_new_chat(self):

        session_dir = os.path.join(DEFAULT_CHAT_SESSIONS_DIR, time.strftime("session_%Y%m%d_%H%M%S"))

        self.chat_store = ChatStore(session_dir)

        if getattr(self.engine, "draft_model", None) is not None:
            self.engine.draft_model.clear()

        self.session_dir_var.set(session_dir)

        self._update_history_status(window_size=None)

        self.log(f"\n=== Neuer Chat (neue Sitzung auf Platte: {session_dir}) ===")



    def _on_archive_chat(self):

        archive_path = filedialog.asksaveasfilename(

            title="Archive chat history losslessly",

            defaultextension=".tar.xz",

            filetypes=[("7z-artiges LZMA-Archiv", "*.tar.xz"), ("Alle Dateien", "*.*")],

            initialfile=os.path.basename(self.chat_store.session_dir) + ".tar.xz",

        )

        if not archive_path:

            return

        try:

            self.chat_store.archive_to(archive_path)

            self.log(f"\n=== Verlauf verlustfrei archiviert: {archive_path} ===")

            messagebox.showinfo("Archived", f"Full history saved to:\n{archive_path}")

        except Exception:

            err = traceback.format_exc()

            self.log("FEHLER beim Archivieren:\n" + err)

            messagebox.showerror("Error", "Archiving failed. See log.")



    def _on_load_archive(self):

        archive_path = filedialog.askopenfilename(

            title="Load chat archive",

            filetypes=[("7z-artiges LZMA-Archiv", "*.tar.xz"), ("Alle Dateien", "*.*")],

        )

        if not archive_path:

            return

        try:

            extract_root = os.path.join(DEFAULT_CHAT_SESSIONS_DIR, "restored_" + time.strftime("%Y%m%d_%H%M%S"))

            restored_dir = ChatStore.load_archive(archive_path, extract_root)

            self.chat_store = ChatStore(restored_dir)

            self.session_dir_var.set(restored_dir)

            self._update_history_status(window_size=None)

            self.log(f"\n=== Archiv geladen, Sitzung wiederhergestellt: {restored_dir} ===")

        except Exception:

            err = traceback.format_exc()

            self.log("FEHLER beim Laden des Archivs:\n" + err)

            messagebox.showerror("Error", "Loading archive failed. See log.")



    def _append_user_turn(self):

        """Legt eine neue User-Nachricht auf der Platte ab. Die

        Datei-Anhaenge werden nur beim allerersten Turn der Sitzung als

        System-Message vorangestellt und ebenfalls persistiert -- nichts

        davon liegt dauerhaft im RAM."""

        prompt = self.prompt_box.get("1.0", "end").strip()

        if not prompt:

            return None

        if len(self.chat_store) == 0 and self.attachments:

            context_parts = []

            for path in self.attachments:

                content = read_attachment(path)

                context_parts.append(f"### Datei: {path}\n{content}")

            system_text = (

                "Dem Nutzer stehen folgende angehaengte Dateien als Kontext zur "

                "Verfuegung. Nutze sie bei der Beantwortung, sofern relevant:\n\n"

                + "\n\n".join(context_parts)

            )

            self.chat_store.append({"role": "system", "content": system_text})



        self.chat_store.append({"role": "user", "content": prompt})

        return prompt



    def _on_generate(self):

        prompt = self._append_user_turn()

        if prompt is None:

            return



        self.gen_btn.config(state="disabled")

        self.stop_event.clear()

        self.stop_btn.config(state="normal")

        self.log(f"\n--- Prompt ---\n{prompt}\n--- Antwort ---")

        self.prompt_box.delete("1.0", "end")



        # Kontextfenster fuer DIESEN Turn aus dem Disk-Verlauf zusammenbauen

        # -- nur dieser Ausschnitt landet im RAM/geht ans Modell, der Rest

        # bleibt unangetastet auf der Platte.

        max_new_tokens = int(self.max_tokens_var.get())

        budget = max(256, self.engine.n_ctx - max_new_tokens - CONTEXT_SAFETY_MARGIN_TOKENS)



        memory_reserve = 0

        memory_text = ""

        if self.memory_enabled_var.get():

            memory_text = self.memory_file.read_all()

            if memory_text:

                memory_reserve = min(

                    self.engine.tokenize_count(memory_text),

                    budget // 3,  # Memory darf max. 1/3 des Fensters belegen

                )



        window = self.chat_store.build_window(budget - memory_reserve, self.engine.tokenize_count)

        window = self._inject_memory(window, memory_text)



        tools_enabled = self.tools_enabled_var.get()

        # OpenSERP-Server-Adresse fuer web_search bekannt machen (leer =

        # DuckDuckGo-HTML-Fallback in _run_web_search). Wird bewusst bei

        # jedem Turn neu gesetzt, damit ein Aendern des Feldes sofort greift.

        set_openserp_base_url(self.openserp_url_var.get())

        tools = build_tool_definitions(

            allow_read=self.tool_read_var.get(),

            allow_list=self.tool_list_var.get(),

            allow_write=self.tool_write_var.get(),

            allow_python=self.tool_python_var.get(),

            allow_diff=self.tool_diff_var.get(),

            allow_browser=self.tool_browser_var.get(),

            allow_http=self.tool_http_var.get(),

            allow_search=self.tool_search_var.get(),

            allow_calculate=self.tool_calculate_var.get(),

            allow_datetime=self.tool_datetime_var.get(),

            allow_append=self.tool_append_var.get(),

            allow_delete=self.tool_delete_var.get(),

            allow_move=self.tool_move_var.get(),

            allow_search_files=self.tool_search_files_var.get(),

            allow_sysinfo=self.tool_sysinfo_var.get(),

        ) if tools_enabled else []



        # System-Erklaerung der aktiven Tools einfuegen -- NACH dem Bauen

        # von 'tools', aber VOR window_start_len, damit dieser Block wie

        # die Memory-Injektion nicht auf die Platte persistiert wird.

        window = self._inject_tools_prompt(window, tools)

        window = self._inject_effort_prompt(window, self.effort_var.get())

        window_start_len = len(window)

        self._update_history_status(window_size=window_start_len)



        def worker():

            try:

                # Batcher statt ein self.after()-Event PRO Token: buendelt
                # alles, was schneller eintrifft als der Tk-Mainloop es
                # abarbeitet, zu einem einzigen Widget-Update. Reduziert die
                # GIL-Konkurrenz zwischen GUI-Thread und Generierungs-Thread
                # spuerbar bei hohen Tokens/Sekunde.
                token_batcher = GuiStreamBatcher(self, self._append_stream)

                def on_token(tok):
                    token_batcher.add(tok)



                reasoning_state = {"started": False}

                reasoning_batcher = GuiStreamBatcher(self, self._append_stream)

                def on_reasoning(tok):
                    prefix = ""
                    if not reasoning_state["started"]:
                        prefix = "\n[Reasoning] "
                        reasoning_state["started"] = True
                    reasoning_batcher.add(prefix + tok)



                def on_tool_log(text):

                    self.after(0, lambda: self.log(text))



                try:

                    max_reasoning_seconds = max(0.0, float(self.max_reasoning_var.get())) * 60

                except ValueError:

                    max_reasoning_seconds = 0

                try:

                    max_tool_rounds = max(1, int(self.tool_rounds_var.get()))

                except ValueError:

                    max_tool_rounds = TOOL_LOOP_MAX_ITERATIONS

                try:

                    repeat_penalty = float(self.repeat_penalty_var.get())

                except ValueError:

                    repeat_penalty = 1.1



                self.engine.run_with_tools(

                    messages=window,

                    tools=tools,

                    max_new_tokens=max_new_tokens,

                    temperature=float(self.temp_var.get()),

                    top_p=float(self.top_p_var.get()),

                    callback=on_token,

                    reasoning_callback=on_reasoning,

                    tool_log=on_tool_log,

                    max_reasoning_seconds=max_reasoning_seconds,

                    max_iterations=max_tool_rounds,

                    repeat_penalty=repeat_penalty,

                    stop_event=self.stop_event,

                )

                # Alles, was waehrend dieses Turns neu entstanden ist

                # (Assistant-Antwort, ggf. Tool-Calls/-Ergebnisse), zurueck

                # auf die Platte schreiben -- damit ist der volle Verlauf

                # weiterhin dort vollstaendig und verlustfrei vorhanden.

                for new_msg in window[window_start_len:]:

                    self.chat_store.append(new_msg)



                self.after(0, lambda: self.log("\n--- Ende ---"))

                self.after(0, lambda: self._update_history_status(window_size=None))

                self._maybe_auto_summarize()

            except Exception:

                err = traceback.format_exc()

                self.log("FEHLER bei der Generierung:\n" + err)

            finally:

                self.after(0, lambda: self.gen_btn.config(state="normal"))

                self.after(0, lambda: self.stop_btn.config(state="disabled"))



        threading.Thread(target=worker, daemon=True).start()



    def _on_stop(self):
        """Bricht eine laufende Generierung so schnell wie moeglich ab --
        z.B. wenn das Modell erkennbar Unsinn labert oder sich verrennt.
        Der eigentliche Abbruch passiert asynchron im Worker-Thread (naechste
        Pruefung von stop_event, spaetestens beim naechsten Stream-Chunk),
        deshalb hier nur das Event setzen und den Button sofort deaktivieren,
        damit kein doppelter Klick moeglich ist."""
        self.stop_event.set()

        self.stop_btn.config(state="disabled")

        self.log("\n[Stop angefordert -- breche sobald moeglich ab...]")

    def _on_effort_change(self, event=None):
        """Effort-Preset gewaehlt -- stellt 'Max. Denkzeit' automatisch auf
        den zur Stufe passenden Wert. Kann danach weiterhin manuell
        ueberschrieben werden, das hier ist nur eine bequeme Voreinstellung."""
        preset = EFFORT_PRESETS.get(self.effort_var.get())

        if preset:
            self.max_reasoning_var.set(str(preset["max_reasoning_minutes"]))

    def _prepend_to_last_message(self, window, block):
        """Haengt einen volatilen Kontext-Block VOR den Inhalt der JUENGSTEN
        Nachricht im Fenster, statt (wie fruehere Versionen) die System-
        Nachricht bei Index 0 zu veraendern.

        Warum das wichtig ist (der eigentliche Geschwindigkeits-Fix hier):
        llama-cpp-python cached intern den zuletzt ausgewerteten Prompt und
        vergleicht bei jedem neuen create_chat_completion()-Aufruf Token
        fuer Token von VORNE, wie weit der neue Prompt mit dem alten
        uebereinstimmt (laengstes gemeinsames Praefix, siehe
        Llama._create_completion/generate -> longest_token_prefix). Nur der
        NICHT uebereinstimmende Rest muss neu durch das Modell laufen
        (Prefill) -- das ist exakt das Prinzip, das vLLM/PagedAttention auf
        Server-Ebene und der "PAT"-Ansatz auf Kernel-Ebene fuer Praefix-
        Wiederverwendung nutzen. Sobald sich aber Token 0 (die System-
        Nachricht) aendert -- und genau das passierte hier vorher bei JEDEM
        Turn, weil Tool-Auswahl/Effort-Stufe/context.md-Inhalt dort mit
        reingemischt wurden -- ist das Praefix ab Token 0 kaputt und der
        GESAMTE bisherige Chatverlauf muss neu prozessiert werden. Bei
        laengeren Unterhaltungen macht genau das die Wartezeit pro Antwort
        mit der Zeit immer laenger (O(n) Prefill pro Turn, O(n^2) ueber die
        Sitzung), obwohl faktisch nur die neue Nutzerfrage wirklich neu ist.

        Die neueste Nachricht muss sowieso bei jedem Turn frisch prozessiert
        werden (sie ist ja neu) -- volatile Zusatzinfos dort statt am Anfang
        einzufuegen kostet also keinerlei zusaetzliches Prefill, haelt aber
        den gesamten Rest des Verlaufs (inkl. System-Nachricht) Byte-fuer-
        Byte stabil und damit fuer den Praefix-Cache wiederverwendbar."""
        if not window:
            return window
        merged_last = dict(window[-1])
        merged_last["content"] = block + "\n\n" + (merged_last.get("content") or "")
        return window[:-1] + [merged_last]

    def _inject_effort_prompt(self, window, effort_key):
        """Fuegt eine kurze Anweisung zur gewaehlten Effort-Stufe ein (wie
        viel Muehe/Denkzeit sich das Modell geben soll). Wird an die
        juengste Nachricht angehaengt (siehe _prepend_to_last_message) statt
        an die System-Nachricht, damit der Rest des Verlaufs Praefix-
        Cache-stabil bleibt. Nicht in chat_store persistiert -- wird bei
        jedem Turn frisch aus der aktuellen Effort-Auswahl gebaut."""
        preset = EFFORT_PRESETS.get(effort_key)
        if not preset:
            return window
        return self._prepend_to_last_message(window, "[" + preset["instruction"] + "]")

    def _inject_memory(self, window, memory_text):
        """Fuegt den aktuellen Inhalt von context.md als Kontext ein --
        an die juengste Nachricht angehaengt (siehe _prepend_to_last_message),
        NICHT mehr in die System-Nachricht gemischt (das wuerde bei jeder
        Aenderung von context.md das Praefix des gesamten Verlaufs invalidieren).
        Wird NICHT in chat_store persistiert, die Datei auf der Platte bleibt
        die einzige Quelle der Wahrheit."""
        if not memory_text:
            return window
        block = (
            "### Projekt-Gedaechtnis (context.md, komprimierter Stand "
            "frueherer Sitzungen/Turns):\n" + memory_text.strip()
        )
        return self._prepend_to_last_message(window, block)

    def _inject_tools_prompt(self, window, tools):
        """Fuegt eine Erklaerung der gerade aktiven Tools ein -- an die
        juengste Nachricht angehaengt (siehe _prepend_to_last_message),
        NICHT mehr in die System-Nachricht gemischt, damit ein An-/Abschalten
        eines Tool-Haekchens zwischen zwei Turns nicht den kompletten
        Praefix-Cache des restlichen Verlaufs zerstoert. Nicht in chat_store
        persistiert -- wird bei jedem Turn frisch aus den aktuell angehakten
        Tool-Checkboxen gebaut, damit sie nie mit dem tatsaechlichen
        Tool-Zustand auseinanderlaeuft."""
        if not tools:
            return window
        block = build_tools_system_prompt(tools)
        if not block:
            return window
        return self._prepend_to_last_message(window, block)



    def _memory_status_text(self):

        n = self.memory_file.size_chars()

        return f"context.md: {n} chars  |  Path: {self.memory_file.path}"



    def _on_show_memory(self):

        text = self.memory_file.read_all() or "(noch leer)"

        win = tk.Toplevel(self)

        win.title("context.md")

        win.geometry("700x500")

        box = scrolledtext.ScrolledText(win, wrap="word")

        box.pack(fill="both", expand=True)

        box.insert("1.0", text)



    def _on_manual_summarize(self):

        threading.Thread(target=self._run_summarize, daemon=True).start()



    def _maybe_auto_summarize(self):

        if not self.memory_enabled_var.get():

            return

        try:

            interval = max(1, int(self.memory_interval_var.get()))

        except ValueError:

            interval = 6

        if self.chat_store.user_turn_count() % interval == 0:

            threading.Thread(target=self._run_summarize, daemon=True).start()



    def _run_summarize(self):

        """Laesst das Modell den aktuellen Fenster-Inhalt zusammenfassen und

        haengt das Ergebnis append-only an context.md an. Laeuft im

        Hintergrund-Thread, blockiert die GUI nicht."""

        try:

            budget = max(256, self.engine.n_ctx - 512 - CONTEXT_SAFETY_MARGIN_TOKENS)

            memory_text = self.memory_file.read_all()

            window = self.chat_store.build_window(budget, self.engine.tokenize_count)

            window = self._inject_memory(window, memory_text)

            summarize_messages = window + [{"role": "user", "content": SUMMARIZE_PROMPT}]



            chunks = []

            summary_batcher = GuiStreamBatcher(self, self._append_stream)



            def on_summary_token(tok):

                # tok kommt aus dem Hintergrund-Thread -- nie direkt Tk-Widgets

                # anfassen, sondern ueber self.after() in den GUI-Thread geben.

                chunks.append(tok)

                summary_batcher.add(tok)



            self.after(0, lambda: self.log("\n[Memory-Update laeuft...]"))

            self.engine.generate_stream(

                messages=summarize_messages,

                max_new_tokens=400,

                temperature=0.2,

                top_p=0.9,

                repeat_penalty=1.1,

                callback=on_summary_token,

            )

            summary = "".join(chunks).strip()

            if summary:

                self.memory_file.append_entry(summary)

                self.after(0, lambda: self.memory_status_label.config(text=self._memory_status_text()))

                self.after(0, lambda: self.log(f"\n[Memory aktualisiert: {len(summary)} Zeichen an context.md angehaengt]"))

        except Exception:

            err = traceback.format_exc()

            self.after(0, lambda: self.log("FEHLER bei Memory-Update:\n" + err))



    def _append_stream(self, token_text):

        self.output_box.insert("end", token_text)

        self.output_box.see("end")



    def _update_history_status(self, window_size):

        total = len(self.chat_store)

        turns = self.chat_store.user_turn_count()

        window_text = str(window_size) if window_size is not None else "-"

        self.history_status_label.config(

            text=f"Stored messages: {total}  |  Chat turns: {turns}  |  In window sent to model: {window_text}"

        )





if __name__ == "__main__":

    app = App()

    app.mainloop()
#!/usr/bin/env python3
"""
LLM Server Launcher - llama.cpp Vulkan

Launcher leve para llama-server com:
- descoberta de GGUF em ~/Apps/LM Studio/models
- descoberta automática do llama-server Vulkan instalado pelo LM Studio
- parâmetros Vulkan/llama.cpp
- speculative decoding: MTP interno, draft externo e n-gram
- início/parada do servidor
- preview do comando real
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Iterable

from PyQt6.QtCore import QProcess, QTimer, Qt, QPointF, QSettings
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QGraphicsDropShadowEffect,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

APP_ROOT = Path.home() / "Apps" / "LM Studio"
MODEL_ROOT = APP_ROOT / "models"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_CTX = 16384
DEFAULT_BATCH = 512
DEFAULT_UBATCH = 128
DEFAULT_PARALLEL = 1
DEFAULT_DRAFT_N_MAX = 2
DEFAULT_DRAFT_P_MIN = 0.0


def version_key(path: Path) -> tuple[int, ...]:
    source = f"{path.parent.name}/{path.name}"
    match = re.search(r"(\d+)\.(\d+)\.(\d+)(?:/llama-server)?$", source)
    if not match:
        match = re.search(r"-(\d+)\.(\d+)\.(\d+)", source)
    if not match:
        return (0,)
    return tuple(int(x) for x in match.groups())


def find_llama_server() -> Path | None:
    explicit = os.environ.get("HERMIS_LLAMA_SERVER")
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit).expanduser())

    candidates.extend([
        APP_ROOT / "llama-server",
        Path.home() / ".lmstudio" / "bin" / "llama-server",
    ])

    backend_root = Path.home() / ".lmstudio" / "extensions" / "backends"
    if backend_root.exists():
        candidates.extend(
            p / "llama-server"
            for p in backend_root.glob("llama.cpp-linux-x86_64-vulkan-avx2-*")
            if p.is_dir()
        )

    which = subprocess.run(
        ["bash", "-lc", "command -v llama-server"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if which:
        candidates.append(Path(which))

    valid = [p for p in candidates if p.is_file() and os.access(p, os.X_OK)]
    if not valid:
        return None

    vulkan = [
        p for p in valid
        if "vulkan" in str(p).lower()
    ]
    return sorted(vulkan or valid, key=version_key)[-1]


def find_gguf_models(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (
            p for p in root.rglob("*.gguf")
            if p.is_file()
            and not p.name.startswith("mmproj")
        ),
        key=lambda p: (p.name.lower(), str(p).lower()),
    )


def is_probable_draft(path: Path) -> bool:
    name = path.name.lower()

    # MTP interno pode aparecer no nome do modelo principal, portanto não
    # tratamos qualquer ocorrência de "mtp" como draft. Já os formatos usados
    # pelos heads externos de Gemma 4 são identificáveis com segurança por:
    #   - "assistant" no nome
    #   - prefixo "mtp-"
    #   - "-mtp-" no nome
    #   - variantes com "_mtp-"
    return (
        "assistant" in name
        or name.startswith("mtp-")
        or "-mtp-" in name
        or "_mtp-" in name
        or "dolphin-draft" in name
        or "-draft-" in name
        or name.startswith("draft-")
    )


def model_label(path: Path) -> str:
    try:
        rel = path.relative_to(MODEL_ROOT)
        return str(rel)
    except ValueError:
        return str(path)


def matching_draft_models(main_model: Path, models: Iterable[Path]) -> list[Path]:
    main = main_model.name.lower()
    tokens = set(re.findall(r"[a-z0-9]+", main))
    candidates: list[tuple[int, Path]] = []

    for path in models:
        if not is_probable_draft(path):
            continue
        name = path.name.lower()
        score = 0

        # Para famílias com tamanho explícito, penaliza fortemente drafts de
        # outro tamanho para evitar escolher, por exemplo, um 26B para um E4B.
        size_tokens = ("e2b", "e4b", "12b", "26b", "31b")
        main_size = next((token for token in size_tokens if token in main), None)
        draft_size = next((token for token in size_tokens if token in name), None)
        if main_size and draft_size and main_size != draft_size:
            continue

        if "gemma-4" in main and "gemma-4" in name:
            score += 20
        if "26b" in main and "26b" in name:
            score += 15
        if "12b" in main and "12b" in name:
            score += 15
        if "31b" in main and "31b" in name:
            score += 15
        if "e4b" in main and "e4b" in name:
            score += 8
        if "mtp" in name:
            score += 10
        if "assistant" in name:
            score += 8

        overlap = sum(1 for token in tokens if len(token) >= 3 and token in name)
        score += min(overlap, 10)

        candidates.append((score, path))

    return [path for _, path in sorted(candidates, key=lambda x: (-x[0], x[1].name.lower()))]


class MotoreLogo(QWidget):
    """Compact geometric mark + wordmark used as the product identity."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(205, 46)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.dark_mode = False

    def set_theme(self, dark: bool) -> None:
        self.dark_mode = dark
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # White rounded-square mark.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(2, 2, 42, 42, 12, 12)

        # Three hollow interlocking cubes: two at the base, one centered above.
        cube_pen = QPen(QColor("#101318"), 1.7)
        cube_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        cube_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(cube_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        def draw_cube(cx: float, cy: float, size: float = 11.0) -> None:
            dx = size * 0.42
            dy = size * 0.25

            top = QPolygonF([
                QPointF(cx, cy - size),
                QPointF(cx + dx, cy - size + dy),
                QPointF(cx, cy - size + (2 * dy)),
                QPointF(cx - dx, cy - size + dy),
            ])
            left = QPolygonF([
                QPointF(cx, cy - size + (2 * dy)),
                QPointF(cx - dx, cy - size + dy),
                QPointF(cx - dx, cy),
                QPointF(cx, cy + dy),
            ])
            right = QPolygonF([
                QPointF(cx, cy - size + (2 * dy)),
                QPointF(cx + dx, cy - size + dy),
                QPointF(cx + dx, cy),
                QPointF(cx, cy + dy),
            ])

            painter.drawPolygon(top)
            painter.drawPolygon(left)
            painter.drawPolygon(right)

        draw_cube(15.5, 31.0)
        draw_cube(30.5, 31.0)
        draw_cube(23.0, 20.0)

        # Wordmark.
        wordmark = QFont("Inter", 20)
        wordmark.setBold(True)
        wordmark.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        painter.setPen(QColor("#f4f7fb" if self.dark_mode else "#17202a"))
        painter.setFont(wordmark)
        painter.drawText(56, 31, "MOTORE")

        painter.end()


class Launcher(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MOTORE")
        self.resize(1050, 820)

        self.settings = QSettings("Motore", "Motore")
        self._restoring_settings = True

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.started.connect(self.on_started)
        self.process.finished.connect(self.on_finished)
        self.process.errorOccurred.connect(self.on_process_error)

        self.all_models: list[Path] = []
        self.server_path: Path | None = None

        # Métricas por solicitação obtidas das linhas de timing emitidas pelo
        # próprio llama-server quando cada slot termina de responder.
        self._server_metric_buffer = ""
        self._pending_request_metrics: dict[str, float | int] | None = None

        self.build_ui()
        self.refresh_everything()
        self.load_settings()
        self._restoring_settings = False
        self.save_settings()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start(1000)

    def build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(28, 24, 28, 24)
        main.setSpacing(18)

        # ---------- Product header ----------
        header = QHBoxLayout()
        header.setContentsMargins(2, 0, 0, 0)
        header.setSpacing(14)

        self.logo_widget = MotoreLogo()
        header.addWidget(self.logo_widget)

        header.addStretch()

        state = QLabel("LOCAL / READY")
        state.setObjectName("headerState")
        header.addWidget(state)

        self.settings_button = QPushButton("CONFIGURAÇÕES")
        self.settings_button.setObjectName("headerButton")
        self.settings_button.clicked.connect(self.open_settings)
        header.addWidget(self.settings_button)
        main.addLayout(header)

        # ---------- 2×2 independent dashboard widgets ----------
        dashboard_grid = QGridLayout()
        dashboard_grid.setHorizontalSpacing(18)
        dashboard_grid.setVerticalSpacing(18)
        dashboard_grid.setColumnStretch(0, 1)
        dashboard_grid.setColumnStretch(1, 1)
        dashboard_grid.setRowStretch(0, 1)
        dashboard_grid.setRowStretch(1, 1)

        self.total_tokens_label = QLabel("—")
        self.generation_speed_label = QLabel("—")
        self.draft_acceptance_label = QLabel("—")

        # Status widget.
        status_card = self._make_card("cardBlue")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(24, 22, 24, 22)
        status_layout.setSpacing(8)

        status_top = QHBoxLayout()
        self.dashboard_status = QLabel("SERVIDOR PARADO")
        self.dashboard_status.setObjectName("cardEyebrow")
        status_top.addWidget(self.dashboard_status)
        status_top.addStretch()
        status_indicator = QLabel("●")
        status_indicator.setObjectName("statusIndicator")
        status_top.addWidget(status_indicator)
        status_layout.addLayout(status_top)

        self.dashboard_message = QLabel("Pronto para iniciar")
        self.dashboard_message.setObjectName("statusTitle")
        status_layout.addWidget(self.dashboard_message)

        self.dashboard_detail = QLabel("Nenhum processo llama-server está em execução.")
        self.dashboard_detail.setObjectName("cardText")
        self.dashboard_detail.setWordWrap(True)
        status_layout.addWidget(self.dashboard_detail)

        self.dashboard_model = QLabel("MODELO // não selecionado")
        self.dashboard_model.setObjectName("cardMeta")
        status_layout.addWidget(self.dashboard_model)
        status_layout.addStretch(1)

        status_actions = QHBoxLayout()
        self.start_button = QPushButton("INICIAR SERVIDOR")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_server)
        self.stop_button = QPushButton("PARAR")
        self.stop_button.setObjectName("secondaryButton")
        self.stop_button.clicked.connect(self.stop_server)
        self.stop_button.setEnabled(False)
        status_actions.addWidget(self.start_button, 2)
        status_actions.addWidget(self.stop_button, 1)
        status_layout.addLayout(status_actions)

        dashboard_grid.addWidget(status_card, 0, 0)
        self._add_shadow(status_card, 28, 6, 70)

        token_card, _ = self._metric_card(
            "TOKENS TOTAIS", self.total_tokens_label, "PROMPT + GERAÇÃO", "cardPurple"
        )
        speed_card, _ = self._metric_card(
            "VELOCIDADE MÉDIA", self.generation_speed_label, "TOKENS / SEGUNDO", "cardTeal"
        )
        draft_card, _ = self._metric_card(
            "DRAFT ACEITO", self.draft_acceptance_label, "ACEITAÇÃO", "cardAmber"
        )

        dashboard_grid.addWidget(token_card, 0, 1)
        dashboard_grid.addWidget(speed_card, 1, 0)
        dashboard_grid.addWidget(draft_card, 1, 1)

        for card in (token_card, speed_card, draft_card):
            self._add_shadow(card, 26, 5, 58)

        main.addLayout(dashboard_grid, 1)

        footer = QHBoxLayout()
        footer.addStretch()
        self.status_label = QLabel("OFFLINE")
        self.status_label.setObjectName("footerStatus")
        footer.addWidget(self.status_label)
        main.addLayout(footer)

        # ---------- Hidden configuration window ----------
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("Configurações — llama.cpp Vulkan")
        self.settings_dialog.resize(1060, 820)

        settings_root = QWidget()
        settings_main = QVBoxLayout(settings_root)
        settings_main.setContentsMargins(20, 20, 20, 20)
        settings_main.setSpacing(14)

        server_box = QGroupBox("Servidor")
        server_grid = QGridLayout(server_box)

        self.server_path_edit = QLineEdit()
        self.server_path_edit.setReadOnly(True)
        browse_server = QPushButton("Escolher…")
        browse_server.clicked.connect(self.choose_server)
        refresh = QPushButton("Atualizar modelos")
        refresh.clicked.connect(self.refresh_everything)

        self.host_edit = QLineEdit(DEFAULT_HOST)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(DEFAULT_PORT)

        server_grid.addWidget(QLabel("llama-server:"), 0, 0)
        server_grid.addWidget(self.server_path_edit, 0, 1)
        server_grid.addWidget(browse_server, 0, 2)
        server_grid.addWidget(refresh, 0, 3)
        server_grid.addWidget(QLabel("Host:"), 1, 0)
        server_grid.addWidget(self.host_edit, 1, 1)
        server_grid.addWidget(QLabel("Porta:"), 1, 2)
        server_grid.addWidget(self.port_spin, 1, 3)
        settings_main.addWidget(server_box)

        model_box = QGroupBox("Modelo")
        model_form = QFormLayout(model_box)
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.draft_combo = QComboBox()
        self.alias_edit = QLineEdit()
        self.alias_edit.setPlaceholderText("opcional")
        model_form.addRow("GGUF principal:", self.model_combo)
        model_form.addRow("Draft externo:", self.draft_combo)
        model_form.addRow("Alias da API:", self.alias_edit)
        settings_main.addWidget(model_box)

        runtime_box = QGroupBox("Runtime / Vulkan")
        runtime_grid = QGridLayout(runtime_box)

        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(512, 1_048_576)
        self.ctx_spin.setValue(DEFAULT_CTX)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(32, 32768)
        self.batch_spin.setValue(DEFAULT_BATCH)
        self.ubatch_spin = QSpinBox()
        self.ubatch_spin.setRange(16, 32768)
        self.ubatch_spin.setValue(DEFAULT_UBATCH)
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 128)
        self.parallel_spin.setValue(DEFAULT_PARALLEL)
        self.gpu_layers = QComboBox()
        self.gpu_layers.addItems(["all", "auto", "0", "1", "10", "20", "30", "40", "50", "60", "80", "100"])
        self.draft_gpu_layers = QComboBox()
        self.draft_gpu_layers.addItems(["all", "auto", "0", "1", "5", "10", "20", "30", "40", "50", "60"])
        self.draft_gpu_layers.setCurrentText("all")
        self.gpu_layers.setCurrentText("all")
        self.flash_attn = QComboBox()
        self.flash_attn.addItems(["auto", "on", "off"])
        self.flash_attn.setCurrentText("on")
        self.k_cache = QComboBox()
        self.k_cache.addItems(["f16", "q8_0", "q4_0", "q4_1", "bf16"])
        self.v_cache = QComboBox()
        self.v_cache.addItems(["f16", "q8_0", "q4_0", "q4_1", "bf16"])
        self.k_cache.setCurrentText("q8_0")
        self.v_cache.setCurrentText("q8_0")
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 64)
        self.threads_spin.setValue(0)
        self.threads_batch_spin = QSpinBox()
        self.threads_batch_spin.setRange(0, 64)
        self.threads_batch_spin.setValue(0)

        self.jinja = QCheckBox("Jinja chat template")
        self.load_mode = QComboBox()
        self.load_mode.addItems(["auto", "mmap", "mlock", "mmap+mlock", "none"])
        self.load_mode.setCurrentText("auto")
        self.no_warmup = QCheckBox("no-warmup")
        self.metrics = QCheckBox("metrics")

        fields = [
            ("Contexto", self.ctx_spin),
            ("Batch", self.batch_spin),
            ("Micro-batch", self.ubatch_spin),
            ("Slots paralelos", self.parallel_spin),
            ("GPU layers", self.gpu_layers),
            ("Flash Attention", self.flash_attn),
            ("KV cache K", self.k_cache),
            ("KV cache V", self.v_cache),
            ("Threads", self.threads_spin),
            ("Threads batch", self.threads_batch_spin),
        ]
        for row, (label, widget) in enumerate(fields):
            runtime_grid.addWidget(QLabel(label + ":"), row // 2, (row % 2) * 2)
            runtime_grid.addWidget(widget, row // 2, (row % 2) * 2 + 1)

        draft_row = QHBoxLayout()
        draft_row.addWidget(QLabel("Draft GPU layers:"))
        draft_row.addWidget(self.draft_gpu_layers)
        draft_row.addStretch()
        runtime_grid.addLayout(draft_row, 5, 0, 1, 4)

        flags = QHBoxLayout()
        flags.addWidget(QLabel("Load mode:"))
        flags.addWidget(self.load_mode)
        for widget in (self.jinja, self.no_warmup, self.metrics):
            flags.addWidget(widget)
        self.jinja.setChecked(True)
        runtime_grid.addLayout(flags, 6, 0, 1, 4)
        settings_main.addWidget(runtime_box)

        spec_box = QGroupBox("Speculative decoding / MTP")
        spec_grid = QGridLayout(spec_box)
        self.spec_combo = QComboBox()
        self.spec_combo.addItems([
            "Desativado",
            "MTP interno (Gemma 4)",
            "MTP externo (Gemma 4)",
            "Draft externo",
            "N-gram",
        ])
        self.spec_combo.currentIndexChanged.connect(self.on_spec_changed)

        self.draft_nmax = QSpinBox()
        self.draft_nmax.setRange(1, 64)
        self.draft_nmax.setValue(DEFAULT_DRAFT_N_MAX)
        self.draft_pmin = QDoubleSpinBox()
        self.draft_pmin.setRange(0.0, 1.0)
        self.draft_pmin.setSingleStep(0.05)
        self.draft_pmin.setValue(DEFAULT_DRAFT_P_MIN)
        self.spec_info = QLabel()
        self.spec_info.setWordWrap(True)

        spec_grid.addWidget(QLabel("Modo:"), 0, 0)
        spec_grid.addWidget(self.spec_combo, 0, 1, 1, 3)
        spec_grid.addWidget(QLabel("N max:"), 1, 0)
        spec_grid.addWidget(self.draft_nmax, 1, 1)
        spec_grid.addWidget(QLabel("P min:"), 1, 2)
        spec_grid.addWidget(self.draft_pmin, 1, 3)
        spec_grid.addWidget(self.spec_info, 2, 0, 1, 4)
        settings_main.addWidget(spec_box)

        app_box = QGroupBox("Aplicativo")
        app_form = QFormLayout(app_box)
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Claro", "light")
        self.theme_combo.addItem("Escuro", "dark")
        self.theme_combo.currentIndexChanged.connect(self.on_theme_changed)
        app_form.addRow("Tema:", self.theme_combo)
        settings_main.addWidget(app_box)

        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMaximumBlockCount(200)
        preview_box = QGroupBox("Comando")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.addWidget(self.command_preview)
        settings_main.addWidget(preview_box, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        log_layout.addWidget(self.log)
        settings_main.addWidget(log_box, 1)

        close_settings = QPushButton("FECHAR CONFIGURAÇÕES")
        close_settings.clicked.connect(self.settings_dialog.hide)
        settings_main.addWidget(close_settings)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings_root)
        settings_layout = QVBoxLayout(self.settings_dialog)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.addWidget(scroll)

        setting_widgets = (
            self.host_edit, self.port_spin, self.model_combo, self.draft_combo,
            self.alias_edit, self.ctx_spin, self.batch_spin, self.ubatch_spin,
            self.parallel_spin, self.gpu_layers, self.draft_gpu_layers,
            self.flash_attn, self.k_cache, self.v_cache, self.threads_spin,
            self.threads_batch_spin, self.jinja, self.load_mode, self.no_warmup,
            self.metrics, self.spec_combo, self.draft_nmax, self.draft_pmin,
        )

        for widget in setting_widgets:
            if hasattr(widget, "currentTextChanged"):
                widget.currentTextChanged.connect(self.update_preview)
            elif hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.update_preview)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self.update_preview)

            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(self.save_settings)
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self.save_settings)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self.save_settings)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self.save_settings)

        self.apply_theme(self.settings.value("app/theme", "light", type=str))


    def apply_theme(self, theme: str) -> None:
        dark = theme == "dark"
        self.logo_widget.set_theme(dark)

        if dark:
            css = """
                QWidget {
                    background: #0c1016;
                    color: #e9eef5;
                    font-family: "Inter", "Noto Sans", sans-serif;
                    font-size: 10pt;
                }
                QLabel { background: transparent; }
                QLabel#headerState {
                    color: #68d8bc; font-size: 8pt; font-weight: 700;
                    letter-spacing: 1.8px;
                }
                QLabel#statusIndicator { color: #68d8bc; font-size: 12pt; }
                QPushButton {
                    background: #171d25; border: 1px solid #28313d;
                    border-radius: 12px; padding: 9px 14px; color: #e6ebf2;
                    font-weight: 600;
                }
                QPushButton:hover { background: #1d2530; border-color: #3a4655; }
                QPushButton:pressed { background: #12171e; }
                QPushButton#headerButton {
                    background: #151c25; border-radius: 10px; padding: 8px 12px;
                    font-size: 8pt; letter-spacing: 1px;
                }
                QPushButton#primaryButton {
                    min-width: 190px; min-height: 46px; border: 1px solid #6aa6f8;
                    border-radius: 15px; color: #f8fbff;
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #245b9e, stop:1 #173f74);
                }
                QPushButton#primaryButton:hover {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #2c6db9, stop:1 #1b4d87);
                }
                QPushButton#secondaryButton {
                    min-width: 190px; min-height: 38px; border: 1px solid #364252;
                    border-radius: 13px; background: #121820; color: #aebbd0;
                }
                QWidget#cardBlue, QWidget#cardPurple, QWidget#cardTeal, QWidget#cardAmber {
                    border: 1px solid #273241; border-radius: 22px; min-height: 175px;
                }
                QWidget#cardBlue {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #19354e, stop:1 #142b3f);
                }
                QWidget#cardPurple {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #33254d, stop:1 #261d3a);
                }
                QWidget#cardTeal {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #173e42, stop:1 #123337);
                }
                QWidget#cardAmber {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #4a3620, stop:1 #382718);
                }
                QLabel#cardEyebrow { color: #93a8c1; font-size: 8pt; font-weight: 700; letter-spacing: 1.5px; }
                QLabel#statusTitle { color: #f5f8fc; font-size: 19pt; font-weight: 750; }
                QLabel#cardText { color: #b3bfd0; font-size: 9pt; }
                QLabel#cardMeta { color: #7f96b0; font-size: 8pt; font-weight: 600; letter-spacing: 0.8px; }
                QLabel#metricLabel { color: #a4b2c4; font-size: 8pt; font-weight: 700; letter-spacing: 1.2px; }
                QLabel#metricValue { color: #f5f8fc; font-size: 25pt; font-weight: 750; }
                QLabel#metricMeta { color: #7f8fa4; font-size: 8pt; letter-spacing: 1px; }
                QLabel#footerStatus { color: #7f8ea3; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }
                QDialog { background: #0c1016; }
                QGroupBox {
                    border: 1px solid #27313d; border-radius: 18px; margin-top: 13px;
                    padding-top: 12px; background: #111720; font-weight: 700;
                }
                QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 7px; color: #98a9bd; }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                    background: #0c1118; border: 1px solid #26303d; border-radius: 10px;
                    padding: 7px; color: #e5ebf3; selection-background-color: #2b5f91;
                }
                QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #4f84b8; }
                QCheckBox { color: #aab8c9; spacing: 7px; }
                QScrollArea { border: none; }
            """
            shadow_alpha = 58
        else:
            css = """
                QWidget {
                    background: #f2f4f7;
                    color: #1d2833;
                    font-family: "Inter", "Noto Sans", sans-serif;
                    font-size: 10pt;
                }
                QLabel { background: transparent; }
                QLabel#headerState {
                    color: #237a69; font-size: 8pt; font-weight: 700;
                    letter-spacing: 1.8px;
                }
                QLabel#statusIndicator { color: #2d9b81; font-size: 12pt; }
                QPushButton {
                    background: #ffffff; border: 1px solid #d5dce4;
                    border-radius: 12px; padding: 9px 14px; color: #26323e;
                    font-weight: 600;
                }
                QPushButton:hover { background: #f8fafc; border-color: #bfc9d5; }
                QPushButton:pressed { background: #eef2f5; }
                QPushButton#headerButton {
                    background: #ffffff; border-radius: 10px; padding: 8px 12px;
                    font-size: 8pt; letter-spacing: 1px;
                }
                QPushButton#primaryButton {
                    min-width: 190px; min-height: 46px; border: 1px solid #4f83c2;
                    border-radius: 15px; color: #ffffff;
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #3679bd, stop:1 #265f9b);
                }
                QPushButton#primaryButton:hover {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #4388ce, stop:1 #2f6fab);
                }
                QPushButton#secondaryButton {
                    min-width: 190px; min-height: 38px; border: 1px solid #cbd4dd;
                    border-radius: 13px; background: #ffffff; color: #526171;
                }
                QWidget#cardBlue, QWidget#cardPurple, QWidget#cardTeal, QWidget#cardAmber {
                    border: 1px solid #d6dde5; border-radius: 22px; min-height: 175px;
                }
                QWidget#cardBlue {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #e2eef9, stop:1 #d4e5f4);
                }
                QWidget#cardPurple {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #eee6f8, stop:1 #e2d7f0);
                }
                QWidget#cardTeal {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #e0f1ec, stop:1 #d1e7e0);
                }
                QWidget#cardAmber {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #f9edd9, stop:1 #f0dfc4);
                }
                QLabel#cardEyebrow { color: #607589; font-size: 8pt; font-weight: 700; letter-spacing: 1.5px; }
                QLabel#statusTitle { color: #1b2a38; font-size: 19pt; font-weight: 750; }
                QLabel#cardText { color: #536474; font-size: 9pt; }
                QLabel#cardMeta { color: #657b8f; font-size: 8pt; font-weight: 600; letter-spacing: 0.8px; }
                QLabel#metricLabel { color: #586b7d; font-size: 8pt; font-weight: 700; letter-spacing: 1.2px; }
                QLabel#metricValue { color: #1b2a38; font-size: 25pt; font-weight: 750; }
                QLabel#metricMeta { color: #718293; font-size: 8pt; letter-spacing: 1px; }
                QLabel#footerStatus { color: #718293; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }
                QDialog { background: #f2f4f7; }
                QGroupBox {
                    border: 1px solid #d6dde5; border-radius: 18px; margin-top: 13px;
                    padding-top: 12px; background: #ffffff; font-weight: 700;
                }
                QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 7px; color: #65778a; }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                    background: #ffffff; border: 1px solid #d0d8e0; border-radius: 10px;
                    padding: 7px; color: #263544; selection-background-color: #bcd6ef;
                }
                QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #5f91bf; }
                QCheckBox { color: #566676; spacing: 7px; }
                QScrollArea { border: none; }
            """
            shadow_alpha = 30

        self.setStyleSheet(css)
        for object_name in ("cardBlue", "cardPurple", "cardTeal", "cardAmber"):
            for widget in self.findChildren(QWidget):
                if widget.objectName() == object_name and widget.graphicsEffect():
                    effect = widget.graphicsEffect()
                    effect.setColor(QColor(0, 0, 0, shadow_alpha))

    def on_theme_changed(self) -> None:
        theme = self.theme_combo.currentData() or "light"
        self.apply_theme(theme)
        self.save_settings()

    def load_settings(self) -> None:
        self._restoring_settings = True

        theme = self.settings.value("app/theme", "light", type=str)
        idx = self.theme_combo.findData(theme)
        self.theme_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.apply_theme(theme)

        self.host_edit.setText(self.settings.value("server/host", DEFAULT_HOST, type=str))
        self.port_spin.setValue(self.settings.value("server/port", DEFAULT_PORT, type=int))

        saved_server = self.settings.value("server/executable", "", type=str)
        if saved_server and Path(saved_server).is_file():
            self.server_path = Path(saved_server)
            self.server_path_edit.setText(saved_server)

        self.alias_edit.setText(self.settings.value("server/alias", "", type=str))
        self.ctx_spin.setValue(self.settings.value("runtime/context", DEFAULT_CTX, type=int))
        self.batch_spin.setValue(self.settings.value("runtime/batch", DEFAULT_BATCH, type=int))
        self.ubatch_spin.setValue(self.settings.value("runtime/ubatch", DEFAULT_UBATCH, type=int))
        self.parallel_spin.setValue(self.settings.value("runtime/parallel", DEFAULT_PARALLEL, type=int))
        self.gpu_layers.setCurrentText(self.settings.value("runtime/gpu_layers", "all", type=str))
        self.draft_gpu_layers.setCurrentText(self.settings.value("runtime/draft_gpu_layers", "all", type=str))
        self.flash_attn.setCurrentText(self.settings.value("runtime/flash_attn", "on", type=str))
        self.k_cache.setCurrentText(self.settings.value("runtime/k_cache", "q8_0", type=str))
        self.v_cache.setCurrentText(self.settings.value("runtime/v_cache", "q8_0", type=str))
        self.threads_spin.setValue(self.settings.value("runtime/threads", 0, type=int))
        self.threads_batch_spin.setValue(self.settings.value("runtime/threads_batch", 0, type=int))
        self.jinja.setChecked(self.settings.value("runtime/jinja", True, type=bool))
        self.load_mode.setCurrentText(self.settings.value("runtime/load_mode", "auto", type=str))
        self.no_warmup.setChecked(self.settings.value("runtime/no_warmup", False, type=bool))
        self.metrics.setChecked(self.settings.value("runtime/metrics", False, type=bool))

        self.spec_combo.setCurrentText(self.settings.value("spec/mode", "Desativado", type=str))
        self.draft_nmax.setValue(self.settings.value("spec/nmax", DEFAULT_DRAFT_N_MAX, type=int))
        self.draft_pmin.setValue(self.settings.value("spec/pmin", DEFAULT_DRAFT_P_MIN, type=float))

        self._saved_model = self.settings.value("server/model", "", type=str)
        self._saved_draft = self.settings.value("server/draft", "", type=str)
        self._saved_server_path = self.settings.value("server/executable", "", type=str)

        geometry = self.settings.value("app/geometry")
        if geometry:
            self.restoreGeometry(geometry)

        self._restoring_settings = False

        if self._saved_model:
            idx = self.model_combo.findData(self._saved_model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
                self.update_draft_models()

        if self._saved_draft:
            idx = self.draft_combo.findData(self._saved_draft)
            if idx >= 0:
                self.draft_combo.setCurrentIndex(idx)

        self.on_spec_changed()
        self.update_preview()

    def save_settings(self) -> None:
        if getattr(self, "_restoring_settings", False):
            return

        self.settings.setValue("app/theme", self.theme_combo.currentData() or "light")
        self.settings.setValue("app/geometry", self.saveGeometry())

        self.settings.setValue("server/executable", str(self.server_path) if self.server_path else "")
        self.settings.setValue("server/host", self.host_edit.text().strip())
        self.settings.setValue("server/port", self.port_spin.value())
        self.settings.setValue("server/model", self.model_combo.currentData() or "")
        self.settings.setValue("server/draft", self.draft_combo.currentData() or "")
        self.settings.setValue("server/alias", self.alias_edit.text())

        self.settings.setValue("runtime/context", self.ctx_spin.value())
        self.settings.setValue("runtime/batch", self.batch_spin.value())
        self.settings.setValue("runtime/ubatch", self.ubatch_spin.value())
        self.settings.setValue("runtime/parallel", self.parallel_spin.value())
        self.settings.setValue("runtime/gpu_layers", self.gpu_layers.currentText())
        self.settings.setValue("runtime/draft_gpu_layers", self.draft_gpu_layers.currentText())
        self.settings.setValue("runtime/flash_attn", self.flash_attn.currentText())
        self.settings.setValue("runtime/k_cache", self.k_cache.currentText())
        self.settings.setValue("runtime/v_cache", self.v_cache.currentText())
        self.settings.setValue("runtime/threads", self.threads_spin.value())
        self.settings.setValue("runtime/threads_batch", self.threads_batch_spin.value())
        self.settings.setValue("runtime/jinja", self.jinja.isChecked())
        self.settings.setValue("runtime/load_mode", self.load_mode.currentText())
        self.settings.setValue("runtime/no_warmup", self.no_warmup.isChecked())
        self.settings.setValue("runtime/metrics", self.metrics.isChecked())

        self.settings.setValue("spec/mode", self.spec_combo.currentText())
        self.settings.setValue("spec/nmax", self.draft_nmax.value())
        self.settings.setValue("spec/pmin", self.draft_pmin.value())
        self.settings.sync()

    def _make_card(self, object_name: str) -> QWidget:
        card = QWidget()
        card.setObjectName(object_name)
        return card

    def _metric_card(self, title: str, value_label: QLabel, meta: str, object_name: str):
        card = self._make_card(object_name)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 18, 22, 20)
        layout.setSpacing(4)
        title_label = QLabel(title)
        title_label.setObjectName("metricLabel")
        value_label.setObjectName("metricValue")
        meta_label = QLabel(meta)
        meta_label.setObjectName("metricMeta")
        layout.addWidget(title_label)
        layout.addSpacing(7)
        layout.addWidget(value_label)
        layout.addSpacing(3)
        layout.addWidget(meta_label)
        return card, layout

    def _add_shadow(self, widget: QWidget, blur: int, y_offset: int, alpha: int) -> None:
        effect = QGraphicsDropShadowEffect(widget)
        effect.setBlurRadius(blur)
        effect.setOffset(0, y_offset)
        effect.setColor(QColor(0, 0, 0, alpha))
        widget.setGraphicsEffect(effect)

    def open_settings(self) -> None:
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def append_log(self, text: str) -> None:
        if text:
            self.log.appendPlainText(text.rstrip())

    def _process_server_metric_lines(self, data: str) -> None:
        """Extrai as métricas da solicitação a partir do print_timings do llama-server."""
        self._server_metric_buffer += data

        lines = self._server_metric_buffer.splitlines(keepends=True)
        complete_lines: list[str] = []
        self._server_metric_buffer = ""

        for line in lines:
            if line.endswith("\n") or line.endswith("\r"):
                complete_lines.append(line)
            else:
                self._server_metric_buffer = line

        for line in complete_lines:
            self._parse_server_metric_line(line)

    def _parse_server_metric_line(self, line: str) -> None:
        prompt_match = re.search(
            r"prompt eval time\s*=\s*[0-9.]+\s*ms\s*/\s*(\d+)\s*tokens.*?"
            r"([0-9.]+)\s*tokens per second",
            line,
            re.IGNORECASE,
        )
        if prompt_match:
            # Se uma nova solicitação começou antes do fechamento tardio da
            # anterior, finalize a anterior antes de iniciar a nova.
            if self._pending_request_metrics is not None and self._pending_request_metrics.get("total_tokens"):
                self._finalize_request_metrics()

            self._pending_request_metrics = {
                "prompt_tokens": int(prompt_match.group(1)),
                "generated_tokens": 0,
                "generation_speed": 0.0,
                "total_tokens": 0,
                "draft_accepted": 0,
                "draft_generated": 0,
            }
            return

        eval_match = re.search(
            r"(?:generation\s+)?eval time\s*=\s*[0-9.]+\s*ms\s*/\s*"
            r"(\d+)\s*(?:tokens|runs).*?([0-9.]+)\s*tokens per second",
            line,
            re.IGNORECASE,
        )
        if eval_match and self._pending_request_metrics is not None:
            self._pending_request_metrics["generated_tokens"] = int(eval_match.group(1))
            self._pending_request_metrics["generation_speed"] = float(eval_match.group(2))
            return

        total_match = re.search(
            r"total time\s*=\s*[0-9.]+\s*ms\s*/\s*(\d+)\s*tokens",
            line,
            re.IGNORECASE,
        )
        if total_match and self._pending_request_metrics is not None:
            self._pending_request_metrics["total_tokens"] = int(total_match.group(1))

            # A linha de aceitação de draft vem logo após o total quando há
            # speculative decoding. Se ela não aparecer, finalize como 0%.
            QTimer.singleShot(100, self._finalize_request_metrics)
            return

        draft_match = re.search(
            r"draft acceptance\s*=\s*[0-9.]+\s*\(\s*(\d+)\s+accepted\s*/\s*"
            r"(\d+)\s+generated",
            line,
            re.IGNORECASE,
        )
        if draft_match and self._pending_request_metrics is not None:
            self._pending_request_metrics["draft_accepted"] = int(draft_match.group(1))
            self._pending_request_metrics["draft_generated"] = int(draft_match.group(2))
            self._finalize_request_metrics()

    def _finalize_request_metrics(self) -> None:
        metrics = self._pending_request_metrics
        if metrics is None:
            return

        total_tokens = int(metrics.get("total_tokens", 0))
        if total_tokens <= 0:
            total_tokens = int(metrics.get("prompt_tokens", 0)) + int(
                metrics.get("generated_tokens", 0)
            )

        generated_tokens = int(metrics.get("generated_tokens", 0))
        generation_speed = float(metrics.get("generation_speed", 0.0))
        draft_generated = int(metrics.get("draft_generated", 0))
        draft_accepted = int(metrics.get("draft_accepted", 0))

        if draft_generated > 0:
            draft_acceptance = draft_accepted / draft_generated * 100.0
            draft_text = f"{draft_acceptance:.2f}% ({draft_accepted}/{draft_generated})"
        else:
            draft_text = "N/A (sem draft)"

        self.total_tokens_label.setText(f"{total_tokens:,}".replace(",", "."))
        self.generation_speed_label.setText(f"{generation_speed:.2f} tok/s")
        self.draft_acceptance_label.setText(draft_text)

        self.append_log(
            "MÉTRICAS DA SOLICITAÇÃO | "
            f"tokens totais: {total_tokens} | "
            f"geração: {generation_speed:.2f} tok/s | "
            f"draft aceito: {draft_text}"
        )

        self._pending_request_metrics = None

    def refresh_everything(self) -> None:
        discovered_server = find_llama_server()
        self.server_path = discovered_server
        self.server_path_edit.setText(str(self.server_path) if self.server_path else "Não encontrado")

        saved_server = getattr(self, "_saved_server_path", "")
        if saved_server and Path(saved_server).is_file():
            self.server_path = Path(saved_server)
            self.server_path_edit.setText(saved_server)
        self.all_models = find_gguf_models(MODEL_ROOT)
        main_models = [p for p in self.all_models if not is_probable_draft(p)]
        self.model_combo.blockSignals(True)
        current = self.model_combo.currentData()
        self.model_combo.clear()

        for path in main_models:
            self.model_combo.addItem(model_label(path), str(path))

        self.model_combo.blockSignals(False)

        preferred_model = current or getattr(self, "_saved_model", "")
        if preferred_model:
            idx = self.model_combo.findData(preferred_model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)

        if self.model_combo.count() and self.model_combo.currentIndex() < 0:
            self.model_combo.setCurrentIndex(0)

        self.update_draft_models()
        self.update_preview()

        if not self.server_path:
            self.append_log("llama-server Vulkan não encontrado.")
            self.append_log("Escolha manualmente o executável ou instale/baixe o backend Vulkan.")

    def choose_server(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar llama-server",
            str(Path.home()),
            "Executável (llama-server);;Todos (*)",
        )
        if path:
            self.server_path = Path(path)
            self.server_path_edit.setText(path)
            self.update_preview()
            self.save_settings()

    def on_model_changed(self) -> None:
        self.update_draft_models()
        self.update_preview()

    def update_draft_models(self) -> None:
        selected = self.current_model()
        self.draft_combo.blockSignals(True)
        current = self.draft_combo.currentData()
        self.draft_combo.clear()
        self.draft_combo.addItem("Automático", "")

        if selected:
            for path in matching_draft_models(selected, self.all_models):
                self.draft_combo.addItem(model_label(path), str(path))

        if current:
            idx = self.draft_combo.findData(current)
            if idx >= 0:
                self.draft_combo.setCurrentIndex(idx)

        self.draft_combo.blockSignals(False)

        if selected and "gemma-4" in selected.name.lower():
            drafts = matching_draft_models(selected, self.all_models)
            if drafts:
                self.draft_combo.setCurrentIndex(1)
                self.spec_combo.setCurrentText("MTP externo (Gemma 4)")

    def current_model(self) -> Path | None:
        value = self.model_combo.currentData()
        return Path(value) if value else None

    def on_spec_changed(self) -> None:
        mode = self.spec_combo.currentText()
        external_mtp = mode == "MTP externo (Gemma 4)"
        external_draft = mode == "Draft externo"
        self.draft_combo.setEnabled(external_mtp or external_draft)
        self.draft_nmax.setEnabled(mode not in {"Desativado", "N-gram"})
        self.draft_pmin.setEnabled(mode not in {"Desativado", "N-gram"})

        messages = {
            "Desativado": "Sem speculative decoding.",
            "MTP interno (Gemma 4)": "Usa --spec-type draft-mtp com a MTP head embutida no modelo principal. Não use --model-draft.",
            "MTP externo (Gemma 4)": "Usa --spec-type draft-mtp + --model-draft com um GGUF Gemma 4 Assistant/MTP externo.",
            "Draft externo": "Usa --spec-type draft-simple + --model-draft para um draft convencional.",
            "N-gram": "Usa --spec-type ngram-mod; não requer modelo draft.",
        }
        info = messages.get(mode, "")
        selected = self.current_model()
        if mode == "MTP interno (Gemma 4)" and selected and "mtp" not in selected.name.lower():
            info += " O GGUF selecionado não indica MTP no nome; confirme que ele realmente contém MTP heads antes de iniciar."
        self.spec_info.setText(info)
        self.update_preview()

    def command_args(self) -> list[str]:
        model = self.current_model()
        if not self.server_path or not model:
            return []

        args = [
            str(self.server_path),
            "--model", str(model),
            "--host", self.host_edit.text().strip() or DEFAULT_HOST,
            "--port", str(self.port_spin.value()),
            "--ctx-size", str(self.ctx_spin.value()),
            "--batch-size", str(self.batch_spin.value()),
            "--ubatch-size", str(self.ubatch_spin.value()),
            "--parallel", str(self.parallel_spin.value()),
            "--gpu-layers", self.gpu_layers.currentText(),
            "--flash-attn", self.flash_attn.currentText(),
            "--cache-type-k", self.k_cache.currentText(),
            "--cache-type-v", self.v_cache.currentText(),
        ]

        alias = self.alias_edit.text().strip()
        if alias:
            args += ["--alias", alias]

        if self.threads_spin.value():
            args += ["--threads", str(self.threads_spin.value())]
        if self.threads_batch_spin.value():
            args += ["--threads-batch", str(self.threads_batch_spin.value())]
        if self.jinja.isChecked():
            args.append("--jinja")
        else:
            args.append("--no-jinja")
        args += ["--load-mode", self.load_mode.currentText()]
        if self.no_warmup.isChecked():
            args.append("--no-warmup")
        if self.metrics.isChecked():
            args.append("--metrics")

        spec = self.spec_combo.currentText()
        if spec == "MTP interno (Gemma 4)":
            args += [
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", str(self.draft_nmax.value()),
                "--spec-draft-ngl", self.draft_gpu_layers.currentText(),
            ]
            if self.draft_pmin.value() > 0:
                args += ["--spec-draft-p-min", f"{self.draft_pmin.value():.2f}"]
        elif spec == "MTP externo (Gemma 4)":
            args += [
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", str(self.draft_nmax.value()),
                "--spec-draft-ngl", self.draft_gpu_layers.currentText(),
            ]
            draft = self.draft_combo.currentData()
            if draft:
                args += ["--model-draft", draft]
            if self.draft_pmin.value() > 0:
                args += ["--spec-draft-p-min", f"{self.draft_pmin.value():.2f}"]
        elif spec == "Draft externo":
            args += [
                "--spec-type", "draft-simple",
                "--spec-draft-n-max", str(self.draft_nmax.value()),
                "--spec-draft-ngl", self.draft_gpu_layers.currentText(),
            ]
            draft = self.draft_combo.currentData()
            if draft:
                args += ["--model-draft", draft]
            if self.draft_pmin.value() > 0:
                args += ["--spec-draft-p-min", f"{self.draft_pmin.value():.2f}"]
        elif spec == "N-gram":
            args += ["--spec-type", "ngram-mod"]

        args += ["--no-ui"]
        return args

    def update_preview(self) -> None:
        args = self.command_args()
        self.command_preview.setPlainText(
            shlex.join(args) if args else "Selecione um llama-server e um GGUF."
        )

    def start_server(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return

        if not self.server_path:
            QMessageBox.critical(self, "llama-server", "Executável llama-server não encontrado.")
            return

        model = self.current_model()
        if not model:
            QMessageBox.critical(self, "Modelo", "Nenhum arquivo GGUF selecionado.")
            return

        args = self.command_args()
        program = args.pop(0)

        if self.spec_combo.currentText() in {"MTP externo (Gemma 4)", "Draft externo"}:
            if not self.draft_combo.currentData():
                QMessageBox.warning(
                    self,
                    "Draft ausente",
                    "Selecione um modelo MTP/draft externo antes de iniciar.",
                )
                return

        self.append_log("INICIANDO:")
        self.append_log(shlex.join([program] + args))

        shader_cache = Path.home() / ".cache" / "mesa_shader_cache"
        shader_cache.mkdir(parents=True, exist_ok=True)
        from PyQt6.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        env.insert("MESA_SHADER_CACHE_DIR", str(shader_cache))
        self.process.setProcessEnvironment(env)

        self.process.setProgram(program)
        self.process.setArguments(args)
        self.process.setWorkingDirectory(str(APP_ROOT))
        self.process.start()

    def stop_server(self) -> None:
        if self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self.append_log("Solicitando encerramento do llama-server…")
        self.process.terminate()
        if not self.process.waitForFinished(2000):
            self.process.kill()

    def read_stdout(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        self._process_server_metric_lines(data)
        self.append_log(data)

    def read_stderr(self) -> None:
        data = bytes(self.process.readAllStandardError()).decode("utf-8", "replace")
        self._process_server_metric_lines(data)
        self.append_log(data)

    def on_started(self) -> None:
        self._server_metric_buffer = ""
        self._pending_request_metrics = None
        self.status_label.setText("ONLINE")
        self.dashboard_status.setText("SERVIDOR INICIADO")
        self.dashboard_message.setText("Servidor local ativo")
        self.dashboard_detail.setText("llama.cpp Vulkan está pronto para receber solicitações.")
        model = self.current_model()
        self.dashboard_model.setText(
            f"MODEL // {model.name}" if model else "MODEL // não selecionado"
        )
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def on_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        self.read_stdout()
        self.read_stderr()
        if self._server_metric_buffer:
            self._parse_server_metric_line(self._server_metric_buffer)
            self._server_metric_buffer = ""
        self._finalize_request_metrics()
        self.status_label.setText(f"OFFLINE / EXIT {code}")
        self.dashboard_status.setText("SERVIDOR PARADO")
        self.dashboard_message.setText("Pronto para iniciar")
        self.dashboard_detail.setText("Nenhum processo llama-server está em execução.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def on_process_error(self, error) -> None:
        self.append_log(f"QProcess error: {error}")
        self.status_label.setText("ERRO")
        self.dashboard_status.setText("FALHA NO SERVIDOR")
        self.dashboard_message.setText("Não foi possível iniciar")
        self.dashboard_detail.setText(str(error))
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def refresh_status(self) -> None:
        running = self.process.state() != QProcess.ProcessState.NotRunning
        if running:
            self.status_label.setText("ONLINE")
            self.dashboard_status.setText("SERVIDOR INICIADO")

    def closeEvent(self, event) -> None:
        self.save_settings()
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(1500):
                self.process.kill()
        event.accept()


def main() -> None:
    app = QApplication([])
    window = Launcher()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

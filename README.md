# ⚡ MOTORE — llama-server Vulkan GUI Launcher

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Qt-PyQt6-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="PyQt6">
  <img src="https://img.shields.io/badge/Vulkan-Supported-E6232E?style=for-the-badge&logo=vulkan&logoColor=white" alt="Vulkan">
  <img src="https://img.shields.io/badge/llama.cpp-Engine-000000?style=for-the-badge" alt="llama.cpp">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License MIT">
</p>

<p align="center">
  <b>MOTORE</b> é uma interface gráfica moderna, leve e otimizada construída em <b>PyQt6</b> para gerenciar e executar o <code>llama-server</code> com aceleração <b>Vulkan</b>, suporte a <b>Speculative Decoding</b> avançado (MTP Gemma 4, Drafts externos e N-gram) e métricas em tempo real.
</p>

---

## 📌 Sumário

- [Visão Geral](#-visão-geral)
- [Recursos Principais](#-recursos-principais)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Suporte a Speculative Decoding](#-suporte-a-speculative-decoding)
- [Parâmetros Configuráveis](#-parâmetros-configuráveis)
- [Licença](#-licença)

---

## 🔍 Visão Geral

O **MOTORE** foi desenvolvido para simplificar a orquestração do backend de inferência local `llama-server`. Ele automatiza a descoberta de executáveis Vulkan, varre diretórios de modelos GGUF (incluindo integração nativa com o diretório do LM Studio) e oferece controle total sobre parâmetros de execução, KV Cache quantizado e otimizações de speculative decoding.

---

## ✨ Recursos Principais

- 🔍 **Descoberta Automática de Servidor e Modelos**:
  - Localiza instalações do `llama-server` (LM Studio, suporte a extensões Vulkan e executáveis no `$PATH`).
  - Varre automaticamente o diretório `~/Apps/LM Studio/models` por arquivos `.gguf`.
  - Associa automaticamente modelos Draft compatíveis ao modelo principal selecionado.

- 🚀 **Suporte Avançado a Speculative Decoding**:
  - **MTP Interno (Gemma 4)**: Utiliza a cabeça MTP (*Multi-Token Prediction*) integrada ao próprio modelo (`--spec-type draft-mtp`).
  - **MTP Externo (Gemma 4)**: Suporte a arquivos GGUF MTP/Assistant dedicados.
  - **Draft Externo**: Suporte a modelos de rascunho convencionais (`--spec-type draft-simple`).
  - **N-gram**: Speculative decoding baseado em padrões locais de n-gram sem necessidade de modelo secundário.

- 📊 **Dashboard de Métricas em Tempo Real**:
  - Leitura do log do servidor com exibição de velocidade de geração (`tok/s`), contagem total de tokens (prompt + geração) e **taxa de aceitação do draft (%)**.

- 🎨 **Interface Moderna e Temática**:
  - Alternância de tema entre **Escuro (Dark)** e **Claro (Light)**.
  - Logotipo e marca geométrica desenhados em vetores nativos Qt.
  - Preview dinâmico em tempo real do comando CLI exato que será executado.

---

## 🛠️ Pré-requisitos

- **Sistema Operacional**: Linux (Fedora / KDE Plasma Wayland ou similar) / Windows / macOS.
- **Drivers de Vídeo**: Suporte a **Vulkan** configurado e funcional.
- **Python**: Versão 3.10 ou superior.
- **llama-server**: Executável compilado com backend Vulkan (fornecido pelo LM Studio ou compilado manualmente).

---

## 📦 Instalação

1. **Clone este repositório**:
   ```bash
   git clone [https://github.com/seu-usuario/motore.git](https://github.com/seu-usuario/motore.git)
   cd motore

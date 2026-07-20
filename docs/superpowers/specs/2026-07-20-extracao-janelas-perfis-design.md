# Design: Extração em janelas candidatas + perfis de deployment (local/cluster)

Data: 2026-07-20
Status: aprovado nas seções 1–3 (aguardando revisão final do documento)

## 1. Contexto e problema

O `parsing-papers` extrai métricas de PDFs científicos via GROBID + LLM (LiteLLM/Ollama).
Dois problemas motivam este redesign:

**Custo computacional.** A extração envia o texto integral do paper (título + abstract +
todas as seções + todas as tabelas) em 2 chamadas sequenciais (dupla extração A/B).
Medições reais nos checkpoints atuais:

| paper | chars do source_text | tokens estimados do prompt | num_ctx enviado |
|---|---|---|---|
| 1-s2.0-...main | 77.272 | ~27.200 | 36.227 |
| CHEN ... Two Banks | 93.133 | ~32.500 | 41.514 |
| Altman 1968 | 47.384 | ~17.200 | 26.265 |

`num_ctx` inflado compete com os pesos do modelo na VRAM de 12GB; não há paralelismo,
batching nem sistema de configuração (tudo é flag de CLI).

**Precisão.** Os 2 papers com maiores prompts (27K e 32K tokens) retornaram 0 registros
em A e B — o modelo pequeno se perde em contexto grande. 8 de 13 linhas atuais estão
flagueadas `needs_review`. A consolidação usa apenas a extração A; B serve só para
flagar divergências, sem resolução.

## 2. Decisões tomadas (com o usuário)

- **Escala**: centenas de PDFs por rodada (100–1.000) — throughput importa.
- **Stack de serving**: vLLM no cluster, Ollama local. LiteLLM abstrai ambos.
- **Corte de tokens**: janelas candidatas selecionadas por heurística + fallback em
  escada até full-text.
- **Reconciliação**: árbitro automático por campo divergente (em vez de flag-only).
- **Avaliação**: sem gabarito rotulado; medir por sinais indiretos (tokens/PDF,
  tempo/PDF, taxa de `needs_review`, divergência A/B, resolução pelo árbitro) sobre
  os 5 PDFs atuais + lista de papers que provavelmente contêm métricas.
- **Arquitetura**: variante A — janelas estáticas single-shot (sem rerank por LLM,
  sem map-reduce).

## 3. Arquitetura

### 3.1 Pipeline de extração em janelas (Seção 1 aprovada)

**Segmentação — novo módulo `blocks.py`.** O `ParsedPaper` vira uma lista de blocos
tipados: cada seção vira um bloco (`heading` + texto); cada tabela vira um bloco
próprio (caption + heading da seção + markdown); título+abstract é bloco fixo sempre
incluído. Cada bloco carrega offsets no `source_text` original para a verificação de
citações continuar válida.

**Seletor heurístico.** Pontua blocos com:
1. catálogo regex de métricas derivado dos 19 campos do schema (AUC, accuracy,
   precision, recall/sensitivity, specificity, F1, RMSE, MAE, R², calibration,
   Brier...);
2. padrões de nomes de modelos (regression, forest, boosting, neural, SVM...);
3. densidade numérica em tabelas;
4. palavras-âncora (performance, validation, discrimination).

Ordena por score e preenche até o orçamento do perfil (estimativa `chars//3`, mesma
convenção do código atual). Orçamento alvo: ~4K tokens (local) / ~6K tokens (cluster),
contra 10–32K hoje.

**Fallback em escada** (garante recall — pior caso = comportamento atual):
seletor com poucos blocos **ou** extração A com `records=[]` → dobra o orçamento e
tenta de novo → ainda vazio → full-text. Após qualquer escalada, A e B rodam sobre o
contexto final escalado (nunca sobre contextos diferentes).

**Dupla extração.** Inalterada em espírito: A (temp 0,1) e B (temp ≥0,4) sobre o
*mesmo* contexto reduzido; pareamento por nome (`fuzz.token_sort_ratio`, mín. 60) e
comparação por campo como já existem. Correção: B passa a herdar `min_num_ctx` de A
(hoje B cai silenciosamente no default 16000).

**Árbitro — novo módulo `arbiter.py`.** Para cada campo divergente: chamada mínima
com os valores candidatos, as quotes de A e B e a janela de texto que contém as
quotes → retorna o valor escolhido. Prompts de poucas centenas de tokens: batelados
pelo vLLM no cluster, e baratos mesmo no Ollama local (roda nos dois perfis).
Falha/invalidação do árbitro → campo volta para flag
`needs_review` (nunca bloqueia o pipeline). A consolidação passa a usar a extração
**reconciliada**, não mais a A crua.

**Verificação de citações.** Passa a rodar sobre as janelas selecionadas (a quote tem
que vir delas), não sobre os ~90K chars completos — mais rápida e igualmente válida.

### 3.2 Perfis de deployment (Seção 2 aprovada)

**Sistema de config** (hoje inexistente): perfis em JSON em
`config/profiles/local.json` e `config/profiles/cluster.json` (zero dependências
novas). Campos: `model`, `api_base`, `provider` (`ollama_chat` / openai-compatível),
`temperature_a`, `temperature_b`, `max_tokens`, `min_num_ctx`,
`window_token_budget`, `max_concurrency`, `llm_timeout_s`, orçamentos da escada de
fallback. CLI ganha `--profile local|cluster`; flags individuais mantêm precedência
sobre o perfil. O menu interativo (`cli.py`) pergunta o perfil em vez de hardcodar
parâmetros.

**Modelos.**
- *Local (12GB)*: mantém `qwen2.5:14b-instruct` Q4_K_M (~9GB) via Ollama. Com prompt
  de ~4K tokens, o KV cache deixa de competir com os pesos.
- *Cluster (>18GB)*: `Qwen2.5-32B-Instruct-AWQ` (~19,5GB de pesos) em vLLM se a
  placa tiver ≥24GB; entre 18–20GB, fallback documentado para
  `Qwen2.5-14B-Instruct-AWQ` (sobra VRAM para KV cache grande e batching). O modelo
  é uma linha no perfil — troca trivial.

**vLLM no cluster.** Serviço opcional no `docker-compose.yml` sob profile `cluster`
(imagem `vllm/vllm-openai`), API OpenAI-compatível; pipeline fala via LiteLLM
provider `openai/` com `api_base` do perfil. Decoding guiado por JSON schema
(xgrammar/outlines) → JSON válido garantido; o pipeline de repair atual vira rede de
segurança. No Ollama local, fluxo atual mantido (json_schema → fallback json_object →
repair).

**Paralelismo** (hoje zero): pool de threads sobre os PDFs — chamadas são HTTP
bloqueante (I/O-bound), threads bastam. Local: 1–2 PDFs simultâneos. Cluster: 8–16
requisições concorrentes; batching contínuo do vLLM resolve o resto. Checkpoints já
são um arquivo por PDF → escrita paralela segura. Triagem e GROBID seguem o mesmo
fluxo, apenas paralelizados.

**Correções incluídas no escopo.**
- B herda `min_num_ctx` de A (`dual_extraction.py`).
- Declarar `typer` no `pyproject.toml` (importado em `cli.py`, não declarado).
- Fallback json_schema→json_object apenas em erro de schema (não mascarar timeout).
- `max_tokens` exposto por perfil (hoje fixo 8000, inacessível via CLI).

### 3.3 Erros, observabilidade e testes (Seção 3 aprovada)

**Tratamento de erros** (princípio: nunca pior que hoje):
- Seletor vazio / extração vazia → escada de fallback até full-text.
- Árbitro falha/inválido → campo volta para `needs_review`.
- vLLM/cluster inalcançável → `doctor.py` estendido valida o perfil ativo
  (conectividade + modelo presente) e falha rápido, em vez de retries de 900s.
- Retries tenacity mantidos para transporte; JSON inválido continua sem re-chamada;
  decoding guiado no cluster praticamente elimina esse caminho.

**Observabilidade** (substitui o gabarito inexistente): checkpoint por PDF registra
tokens estimados por chamada, tempo por etapa, orçamento de janela usado, nível de
fallback atingido e decisões do árbitro. Novo comando
`parsing-papers compare <run_a> <run_b>` resume: tokens/PDF, tempo/PDF,
registros/paper, taxa de `needs_review`, taxa de divergência A/B, taxa de resolução
pelo árbitro.

**Testes** (estilo existente: determinísticos, sem GPU):
- Unitários: segmentação em blocos (TEI fixture), ranking do seletor (bloco com
  métrica > enchimento), preenchimento de orçamento, escada de fallback (extrator
  mock vazio), decisão/merge do árbitro, carregamento de perfil e precedência CLI >
  perfil.
- Integração com servidor LLM mock (padrão já usado nos testes atuais).
- Fixture nova: TEI sintético grande — hoje nenhum fixture representa os TEIs reais
  de 50–150KB.
- Os 69 testes atuais devem continuar verdes; o fallback garante compatibilidade.

## 4. Não-objetivos (YAGNI)

- Não alterar a triagem PRISMA (já é barata: título+abstract).
- Não alterar a integração `registry.py` (fora do escopo, conforme o usuário).
- Sem RAG/embeddings, sem rerank por LLM, sem map-reduce por bloco.
- Sem re-chamada de LLM em JSON inválido (caminho raro com decoding guiado).
- Sem refatorações além das 4 correções listadas em 3.2.

## 5. Critérios de sucesso (medidos com `compare`)

- Tokens de prompt/PDF: de 10–32K para ≤6K (nos papers sem fallback).
- Os 2 papers que hoje retornam 0 registros passam a extrair ≥1 registro (via janela
  ou fallback).
- Taxa de `needs_review` menor que a atual (8/13 linhas) após reconciliação.
- Tempo/PDF menor no perfil local; throughput do perfil cluster ≥4× o local.
- 69 testes atuais verdes + novos testes dos módulos novos.

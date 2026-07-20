# parsing-papers

Pipeline de extração de metadados de papers científicos (PDF) para a meta-análise
sobre desempenho de modelos de ML em crédito agrícola. Implementa as etapas
propostas: **triagem PRISMA de elegibilidade**, parsing determinístico (GROBID),
extração com schema forçado e citação obrigatória, verificação automática de
citações, dupla extração (espírito PRISMA), checagem de sanidade e amostragem
para auditoria humana.

Este pipeline extrai apenas os campos metodológicos e de desempenho da planilha
(`Model_used` ... `Standard_error`). Os campos `doi`, `Year`, `Author`,
`Journal`, `Country`, `Latitude`, `Longitude` são extraídos por outro método e
entram como colunas vazias na saída, para merge posterior — a menos que você
use a integração com o registry compartilhado (ver seção "Integração com
SPE/pontodoi" abaixo), caso em que essas colunas são preenchidas automaticamente.

## Início rápido (menu interativo)

Se você não tem intimidade com Docker/GROBID/Ollama ou prefere não decorar
flags de linha de comando, use o menu interativo em vez das seções técnicas
abaixo:

```bash
pip install -e .
docker compose up -d
docker exec -it parsing_papers_ollama ollama pull qwen2.5:14b-instruct

parsing-papers
```

Isso abre um menu colorido (mesmo estilo do SPE e do pontodoi) com:

1. **Verificar ambiente** — checa Docker, os dois containers, GROBID e Ollama
   respondendo, e se o modelo já foi baixado, nessa ordem. Se algo estiver
   faltando, mostra o comando exato para resolver — não tenta corrigir nada
   sozinho.
2. **Rodar sobre uma pasta de PDFs** — pergunta a pasta de entrada/saída e o
   modelo, com valores padrão prontos (Enter aceita o padrão).
3. **Rodar via registry compartilhado** — para quem usa o Synoptic Paper
   Engine + pontodoi (ver seção "Integração com SPE/pontodoi").
4. **Reconsolidar planilha** — reconstrói a planilha final a partir de
   checkpoints já existentes, sem reprocessar PDFs.

Ao final de cada rodada, um resumo em linguagem simples explica quantos
papers foram processados, quantos precisam de revisão humana (e por quê) e
onde estão os arquivos — sem precisar entender de antemão o que significam
colunas como `needs_review` ou `dual_extraction_diverges`.

`parsing-papers doctor` roda só o diagnóstico de ambiente, fora do menu —
útil para checar rapidamente se algo mudou (ex: reiniciou o Docker Desktop).

As seções abaixo (Setup, Uso) documentam o mesmo pipeline por linha de
comando com flags explícitas — útil para automação/scripts, ou para quem
quer controle fino sobre cada parâmetro.

## Arquitetura

```
src/parsing_papers/
  screening_schema.py  # Etapa 0: schema da triagem PRISMA (decisão + critérios I/E/M/ME)
  screening_prompts.py  # Etapa 0: prompt com os critérios formais de inclusão/exclusão
  screening.py            # Etapa 0: aplica a triagem via LLM, com citação obrigatória
  schema.py           # JSON schema (Pydantic) com EvidenceField (valor + citação obrigatória)
  grobid_client.py     # Etapa 1: parsing GROBID -> seções rotuladas
  prompts.py            # Prompt de extração (regras anti-alucinação)
  llm_client.py          # Etapa 2: extração via LiteLLM (Ollama por padrão)
  dedup.py                # Etapa 2: deduplica modelo repetido na mesma extração
  verification.py         # Etapa 3: fuzzy match citação x texto fonte (rapidfuzz)
  dual_extraction.py       # Etapa 4: roda 2x, pareia por nome e compara divergências
  numparse.py               # Parser numérico único (decimal/milhar PT-BR e EN)
  sanity_checks.py          # Etapa 5: regras de faixa (AUC 0.5-1, proporções 0-1, etc)
  audit_sampling.py          # Etapa 6: amostra 15-20% para revisão humana
  consolidate.py               # Monta a tabela final e exporta xlsx/csv
  pipeline.py                    # CLI orquestrador com checkpointing (flags explícitas)
  registry.py                     # Leitura/escrita do registry.jsonl compartilhado (SPE/pontodoi)
  doctor.py                        # Diagnóstico de ambiente (Docker/GROBID/Ollama/modelo)
  cli.py                            # Menu interativo (comando `parsing-papers`)
  ui.py                              # Camada de apresentação do menu (cores, tabelas, prompts)
```

## Triagem PRISMA (etapa 0)

Antes de qualquer extração de métricas, cada PDF passa por uma triagem de
elegibilidade baseada nos critérios formais do documento *Critérios de
Inclusão e Exclusão para Revisão Sistemática e Metanálise* (aplicação a
modelos de IA/ML/DL em risco de crédito). O LLM lê título+abstract (com
fallback para o texto completo se o abstract vier vazio) e decide, com
citação obrigatória por critério aplicado — mesmo mecanismo anti-alucinação
usado na extração de métricas — uma das três categorias:

| Decisão | Significado |
|---|---|
| `include_review_and_metaanalysis` | Trata de risco de crédito, aplica modelo e possui métricas quantitativas extraíveis por modelo |
| `include_review_only` | Relevante para a revisão, mas sem dados suficientes/comparáveis para a metanálise |
| `exclude` | Não atende aos critérios formais, temáticos ou metodológicos |

Papers com decisão `exclude` **não prosseguem** para as etapas de extração
(2 em diante) — a chamada de LLM mais cara do pipeline (dupla extração com
texto completo) só roda para papers já triados como elegíveis.

**Critérios de inclusão na revisão (I1-I8):** recorte temporal (2000-2025),
idioma (inglês/português/espanhol), artigo científico, foco em risco de
crédito/default/credit scoring, uso de modelo quantitativo/ML/DL, ao menos
uma métrica de desempenho, aplicação empírica (não só conceitual), e
informações mínimas para extração.

**Critérios de exclusão da revisão (E1-E11):** fora do recorte temporal ou
idioma, é revisão/bibliometria/tese/relatório (não é artigo empírico
primário), não trata de risco de crédito, trata de outro tipo de risco,
trata de crédito sem foco em risco, é puramente teórico, menciona IA/ML/DL
apenas perifericamente, sem métricas reportadas, duplicado, ou sem
informação mínima.

**Critérios adicionais para a metanálise (M1-M7):** modelo claramente
identificado, métrica comparável (AUC/Accuracy/F1/Recall/Specificity),
métrica associada a um modelo específico, valor numérico informado, medida
de dispersão (desejável), estratégia de validação informada, e resultados
extraíveis separadamente por modelo quando há múltiplos modelos no estudo.

**Exclusão específica da metanálise (ME1-ME6):** o estudo pode permanecer
na revisão sistemática mesmo excluído da metanálise — por exemplo, sem
valor numérico recuperável, métrica sem associação clara ao modelo, ou
resultado só em gráfico.

Os códigos completos e suas justificativas estão em `screening_schema.py`.

## Setup

### 1. Dependências Python

```bash
pip install -r requirements.txt
```

### 2. Serviços (GROBID + Ollama)

```bash
docker compose up -d
```

Isso sobe:
- **GROBID** em `http://localhost:8070` (parsing de PDF -> TEI XML)
- **Ollama** em `http://localhost:11434` (LLM local para extração)
- **vLLM** (opcional, perfil `cluster`) em `http://localhost:8000` — não sobe
  no `up` padrão; suba explicitamente com `docker compose --profile cluster up -d vllm`

Baixe um modelo no Ollama antes de rodar o pipeline. **A escolha do modelo
depende da VRAM da sua GPU** — um paper científico completo gera prompts de
6.000-20.000+ tokens (o pipeline manda o texto completo, não um resumo), e o
KV-cache desse contexto consome VRAM além dos pesos do modelo:

```bash
# ~9GB em Q4 -- cabe com folga em GPUs de 12GB (ex: RTX 3060), padrão do projeto
docker exec -it parsing_papers_ollama ollama pull qwen2.5:14b-instruct

# ~5GB em Q4 -- mais rápido, cabe em GPUs de 8GB ou menos
docker exec -it parsing_papers_ollama ollama pull llama3.1:8b

# ~20GB em Q4 -- só cabe inteiro em GPUs de 24GB+; em GPUs menores o Ollama
# faz offload parcial para CPU e fica MUITO mais lento com contexto grande
# (pode facilmente estourar timeouts de 600s)
docker exec -it parsing_papers_ollama ollama pull qwen2.5:32b-instruct
```

Modelos maiores tendem a seguir melhor o schema JSON e citar com mais
fidelidade, mas só valem a pena se couberem inteiros na VRAM disponível —
rodar em modo híbrido GPU+CPU com contexto grande é impraticavelmente lento
para uso em lote. Confira sua VRAM com `nvidia-smi` antes de escolher.

Desde a versão com janelas candidatas, o prompt de extração caiu de 10–32K
para ~4–6K tokens; o gargalo de `num_ctx` em GPUs de 12GB foi eliminado.
Para cluster (>18GB), use o perfil `cluster`: `Qwen2.5-32B-Instruct-AWQ` em
vLLM (GPU ≥24GB; para 18–20GB, `Qwen2.5-14B-Instruct-AWQ`).

### 2.1 Habilitar GPU no Ollama (Docker + WSL2/Linux)

O `docker-compose.yml` já vem configurado para usar GPU NVIDIA. Requisitos:

1. Driver NVIDIA para WSL2 instalado no Windows (não dentro do WSL).
2. Testar se o Docker enxerga a GPU: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`
3. Confirmar que o container está usando a GPU: `docker exec -it parsing_papers_ollama nvidia-smi`

Sem isso, o Ollama roda 100% em CPU — funciona, mas processar contextos
grandes fica lento o suficiente para inviabilizar lotes de vários papers.

### 3. Colocar os PDFs

Coloque os PDFs em `data/pdfs/`. **Não é mais necessário triar manualmente
antes** — o pipeline aplica os critérios de inclusão/exclusão automaticamente
na etapa 0 (ver seção "Triagem PRISMA" acima) e só manda para a extração de
métricas os papers elegíveis. Se a triagem já foi feita manualmente e
`data/pdfs/` só contém PDFs já aprovados, use `--skip-screening` para pular
essa etapa.

## Uso

Rodar o pipeline completo:

```bash
python -m parsing_papers.pipeline run \
  --pdf-dir data/pdfs \
  --out-dir data/extracted \
  --profile local \
  --citation-threshold 90
```

`--profile local|cluster` seleciona o conjunto de parâmetros de
`config/profiles/<nome>.json` (modelo, endpoint, orçamento de janela,
concorrência). Flags `--model`, `--api-base`, `--temperature`,
`--llm-timeout-s`, `--min-num-ctx` agora são overrides individuais do perfil.
Exemplo no cluster:

```bash
python -m parsing_papers.pipeline run --pdf-dir data/pdfs --out-dir data/extracted --profile cluster
```

Isso gera, em `data/extracted/`:
- `screening/<paper_id>.json` — resultado da triagem PRISMA (etapa 0): decisão,
  critérios aplicados com evidência, justificativa. Papers com `exclude` não
  têm checkpoint de extração.
- `triagem_prisma.xlsx` / `.csv` — planilha de triagem: uma linha por paper
  (incluído ou não), com decisão, códigos de critério em cada categoria
  (I/E/M/ME) e justificativa — a "Categoria recomendada para a planilha de
  triagem" da proposta metodológica.
- `checkpoints/<paper_id>.json` — resultado bruto por paper (extração dupla +
  verificações + comparações), só para papers elegíveis. Permite retomar sem
  reprocessar: rodar `run` de novo pula papers com checkpoint (`--force` para
  reprocessar).
- `checkpoints/<paper_id>.partial.json` — checkpoint **intermediário**,
  salvo assim que a extração A termina (antes da extração B começar) e
  removido ao concluir o checkpoint final do paper. Se o processo for
  interrompido entre as duas chamadas ao LLM (timeout, crash do Ollama,
  queda de energia), reprocessar reaproveita a extração A já feita em vez
  de refazer as duas chamadas do zero.
- `tei_cache/<paper_id>.tei.xml` — saída bruta do GROBID, cacheada.
- `metadados_extraidos.xlsx` / `.csv` — a planilha final, uma linha por
  modelo, com colunas de proveniência/QA (`citation_verified`,
  `sanity_failed`, `dual_extraction_diverges`, `needs_review`, etc).
- `auditoria_amostra.xlsx` / `.csv` — amostra de ~18% dos registros para
  revisão manual (prioriza os que têm algum flag; completa com sorteio
  aleatório de registros "limpos" para calibrar taxa de erro). Tem uma
  coluna `human_verdict` para o revisor preencher.

Use `--skip-screening` para pular a etapa 0 e mandar todos os PDFs de
`--pdf-dir` direto para a extração (útil se a triagem já foi feita
manualmente fora do pipeline).

Se quiser só reconsolidar as planilhas a partir de checkpoints já existentes
(sem reprocessar PDFs):

```bash
python -m parsing_papers.pipeline consolidate --out-dir data/extracted
```

## Comparando rodadas (benchmark)

`python -m parsing_papers.pipeline compare --run-a <out_dir_baseline> --run-b <out_dir_novo>`
imprime tokens/prompt, registros, divergências, resoluções do árbitro e tempo
por paper, lado a lado.

## Integração com SPE/pontodoi

Se a lista de papers vem de uma revisão sistemática feita no **Synoptic
Paper Engine** (busca + triagem PRISMA) e os PDFs foram baixados pelo
**pontodoi**, os dois exportam/mantêm um `registry.jsonl` compartilhado —
um arquivo com um registro por paper e três status independentes
(`metadata_status`, `fulltext_status`, `extraction_status`). Em vez de
copiar PDFs manualmente para `data/pdfs/`, rode:

```bash
python -m parsing_papers.pipeline registry-run \
  --registry /caminho/para/registry.jsonl \
  --out-dir data/extracted \
  --model ollama_chat/qwen2.5:14b-instruct
```

(ou, pelo menu interativo: opção 3, "Rodar via registry compartilhado")

Isso filtra automaticamente só os papers com `fulltext_status=done` e
`extraction_status` pendente/falho, copia os PDFs correspondentes para uma
pasta interna, roda o pipeline normal, e ao final: preenche as colunas
`doi`/`Year`/`Author`/`Journal`/`Country`/`Latitude`/`Longitude` da planilha
final automaticamente a partir dos metadados do SPE (em vez de ficarem
vazias, como no modo `run` isolado), e grava `extraction_status` de volta
no registry — `done` para quem foi extraído, `not_applicable` para quem foi
excluído na triagem PRISMA (etapa 0), `failed` para erros. Rodar de novo só
reprocessa o que ainda estiver pendente/falho.

## Colunas de QA na saída

Além das colunas da planilha oficial, cada linha traz:

| Coluna | Significado |
|---|---|
| `citation_verified` | `True` se todas as citações do registro bateram no fuzzy match |
| `unverified_fields` | Lista dos campos cuja citação não verificou |
| `sanity_failed` | `True` se algum valor numérico violou uma regra de faixa |
| `sanity_messages` | Detalhe das violações de faixa |
| `dual_extraction_diverges` | `True` se restou divergência entre as duas extrações NÃO resolvida pelo árbitro automático, ou se o registro não teve par correspondente na outra extração |
| `dual_model_used_b` | Nome do modelo pareado na extração B (pareamento por similaridade de `model_used`, não por posição) |
| `dual_match_score` | Score de similaridade (0-100) do pareamento A/B pelo nome do modelo |
| `dual_divergent_fields` | Lista dos campos que divergiram entre as duas extrações e NÃO foram resolvidos pelo árbitro, para essa linha |
| `dual_unmatched` | `True` se o registro de A não encontrou correspondente em B (score de similaridade abaixo do limiar) |
| `arbiter_resolved_fields` | Lista dos campos divergentes que o árbitro automático resolveu (escolha "a" ou "b"), para essa linha |
| `needs_review` | OR de todos os flags acima (divergência só conta se NÃO resolvida pelo árbitro) — usado para priorizar a amostra de auditoria |

Recomenda-se **não aceitar automaticamente** nenhuma linha com
`needs_review=True` na planilha final da meta-análise sem revisão humana.

## Testes

O repositório tem 107 testes unitários cobrindo os módulos determinísticos
(schema, parsing GROBID, verificação de citação, sanidade, parser numérico,
deduplicação, pareamento A/B, reconciliação na planilha final, checkpoint
intermediário, triagem PRISMA, consolidação, amostragem) — nenhum deles
depende de GROBID/Ollama estarem rodando:

```bash
pip install pytest
python -m pytest tests/ -v
```

Um teste (`test_consolidate_and_audit.py`) simula uma citação alucinada
(inventada) e confirma que ela é corretamente marcada como não verificada e
sinalizada para revisão — validando o mecanismo central de anti-alucinação
do pipeline.

O teste end-to-end real (PDF -> GROBID -> LLM -> planilha) não roda neste
ambiente porque depende de Docker (GROBID) e de um modelo Ollama baixado
localmente (vários GB). Para validar end-to-end na sua máquina:

```bash
docker compose up -d
docker exec -it parsing_papers_ollama ollama pull llama3.1:8b   # modelo leve p/ teste rápido
# coloque 1 PDF de teste em data/pdfs/
python -m parsing_papers.pipeline run --pdf-dir data/pdfs --out-dir data/extracted --model ollama_chat/llama3.1:8b
```

Depois, abra `data/extracted/metadados_extraidos.xlsx` e confira se os
modelos/métricas foram extraídos corretamente e se as colunas de QA fazem
sentido (comparando manualmente com o PDF).

## Notas de design / decisões

- **Triagem gate a extração, mas nunca decide "excluir" no silêncio.** Se a
  triagem falhar (erro de LLM, JSON malformado que nem o auto-reparo resolve),
  o paper não é automaticamente excluído — fica sem checkpoint de triagem e
  sem checkpoint de extração, para ser revisado manualmente, em vez de ser
  descartado por uma falha técnica. Já quando o LLM retorna uma `decision`
  fora do vocabulário conhecido e não mapeável para um sinônimo comum, o
  auto-reparo default para `exclude` (com aviso em `screening_warnings`) —
  o raciocínio é que incluir algo incerto silenciosamente é pior do que
  excluir e deixar rastreável para revisão.
- **Uma linha por modelo, não por paper.** Um paper que testa 10 modelos
  gera 10 `ModelRecord` (etapa 4 da proposta original). Modelos repetidos
  dentro da mesma extração (ex: o mesmo modelo citado em métodos e de novo
  em resultados) são deduplicados automaticamente antes de entrar na
  planilha (`dedup.py`) — só são fundidos quando nome e métricas coincidem;
  nomes parecidos com métricas diferentes são mantidos separados.
- **Pareamento A/B por nome, não por posição.** A comparação da dupla
  extração (etapa 4/PRISMA) pareia os registros de A e B por similaridade de
  `model_used` (fuzzy match), não pela ordem em que aparecem — isso evita
  divergência espúria quando o LLM lista os mesmos modelos em ordens
  diferentes entre as duas chamadas. Registros sem correspondente do outro
  lado ficam marcados como `dual_unmatched` para revisão humana.
- **Árbitro automático de divergências.** As divergências da dupla extração
  vão para um árbitro automático (uma terceira chamada mínima ao LLM, que
  decide "a"/"b"/"neither" campo a campo); as que ele não resolve continuam
  flagadas para revisão humana.
- **Baseline.** Regressão Logística é marcada `baseline=yes` quando usada
  como controle, conforme convenção do projeto; isso é reforçado no prompt
  (regra 4), mas cabe ao revisor humano confirmar nos casos ambíguos.
- **`response_format=json_schema`** é tentado primeiro (structured output
  estrito); se o modelo/servidor Ollama não suportar, cai automaticamente
  para `json_object` + validação Pydantic com recuperação de JSON
  malformado (`_try_recover_json`).
- **Custo de troca de LLM.** Como tudo passa por LiteLLM, trocar de modelo
  local (Ollama) para uma API (Groq, Together, OpenAI etc.) é só mudar
  `--model`/`--api-base` — nenhum código muda.

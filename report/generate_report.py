from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Frame,
    KeepTogether,
)


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "LLM_Routing_Minor_Project_Report.pdf"


class NumberedDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        self.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=self.footer)])

    def footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"{doc.page}")
        canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=18,
    )
)
styles.add(
    ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        fontSize=12,
        leading=17,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#333333"),
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="Heading1Report",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=12,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="Heading2Report",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#244062"),
        spaceBefore=8,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="BodyReport",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.6,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="SmallReport",
        parent=styles["BodyText"],
        fontSize=8.2,
        leading=11,
        spaceAfter=3,
    )
)
styles.add(
    ParagraphStyle(
        name="TOC",
        parent=styles["BodyText"],
        fontSize=10,
        leading=16,
        leftIndent=8,
    )
)
styles.add(
    ParagraphStyle(
        name="Caption",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceBefore=3,
        spaceAfter=8,
    )
)


def P(text, style="BodyReport"):
    return Paragraph(text, styles[style])


def section(title):
    return P(title, "Heading1Report")


def subsection(title):
    return P(title, "Heading2Report")


def table(data, widths=None, font=7.8):
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font),
                ("LEADING", (0, 0), (-1, -1), font + 2),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A6A6A6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF1F8")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def figure(filename, caption, width=16.5 * cm):
    path = ROOT / "figures" / filename
    if not path.exists():
        return P(f"Figure unavailable: {filename}", "SmallReport")
    image = Image(str(path), width=width, height=width * 0.39)
    return KeepTogether([image, P(caption, "Caption")])


story = []

story += [
    Spacer(1, 1.2 * cm),
    P("MINOR PROJECT REPORT", "Subtitle"),
    P("Cost-Aware Routing of Large Language Models", "ReportTitle"),
    P(
        "An empirical study of confidence, difficulty, self-consistency, and learned routing strategies for mathematical question answering",
        "Subtitle",
    ),
    Spacer(1, 1.0 * cm),
    P("<b>Submitted by</b>", "Subtitle"),
    P("[STUDENT NAME] &nbsp;&nbsp; | &nbsp;&nbsp; [ROLL NUMBER]", "Subtitle"),
    Spacer(1, 0.8 * cm),
    P("<b>Under the guidance of</b>", "Subtitle"),
    P("[SUPERVISOR NAME]", "Subtitle"),
    Spacer(1, 1.0 * cm),
    P("[DEPARTMENT NAME]", "Subtitle"),
    P("[INSTITUTION NAME]", "Subtitle"),
    P("[ACADEMIC YEAR]", "Subtitle"),
    Spacer(1, 1.5 * cm),
    P(
        "<b>Project repository:</b> llm-routing-research<br/>"
        "<b>Primary dataset:</b> OpenAI GSM8K<br/>"
        "<b>Execution backend:</b> Local Ollama models",
        "Subtitle",
    ),
    PageBreak(),
]

story += [
    section("CERTIFICATE"),
    P(
        "This is to certify that the project report entitled <b>“Cost-Aware Routing of Large Language Models”</b> "
        "is a record of the minor project work carried out by <b>[STUDENT NAME]</b>, [ROLL NUMBER], under my "
        "supervision and guidance during the academic year [ACADEMIC YEAR]."),
    Spacer(1, 1.3 * cm),
    P("Supervisor: ______________________________", "Subtitle"),
    P("Head of Department: ______________________", "Subtitle"),
    Spacer(1, 1.0 * cm),
    P("Date: ____________________ &nbsp;&nbsp;&nbsp; Place: ____________________", "Subtitle"),
    PageBreak(),
    section("DECLARATION"),
    P(
        "I declare that this report is my original work, prepared for the minor project requirement. The experimental "
        "results presented here were generated from the project’s recorded model outputs and offline replay pipeline. "
        "All external concepts and sources used in the report are acknowledged in the references."),
    Spacer(1, 1.5 * cm),
    P("[STUDENT NAME]", "Subtitle"),
    P("[ROLL NUMBER]", "Subtitle"),
    PageBreak(),
    section("ACKNOWLEDGEMENT"),
    P(
        "I express my sincere gratitude to my project supervisor, [SUPERVISOR NAME], for guidance and feedback "
        "throughout this work. I also thank [INSTITUTION NAME] and the Department of [DEPARTMENT NAME] for providing "
        "the academic environment and computing resources required for the study. Finally, I acknowledge the open "
        "source communities behind GSM8K, Ollama, the evaluated language models, and the supporting Python libraries."),
    PageBreak(),
]

story += [
    section("ABSTRACT"),
    P(
        "Large language models vary substantially in accuracy, latency, and computational cost. Using a strong model "
        "for every request can improve answer quality, but it is inefficient when a smaller model is already capable "
        "of solving the question. This project investigates whether a router can selectively escalate questions from a "
        "small local model to a stronger model. The study uses 25 pilot questions, a disjoint 50-question calibration "
        "set, and a 100-question GSM8K evaluation set. It compares random escalation, input-difficulty routing, "
        "self-consistency agreement, token log-probability confidence, capability-profile routing, two-stage and "
        "three-tier cascades, and a local RouteLLM BERT router."),
    P(
        "The experiments record correctness, generation time, token counts, sampled answers, and routing overhead. "
        "Results show a clear quality-cost frontier: small-only routing achieves 71% and 54% accuracy for the two "
        "small-model pairs, while strong-only routing achieves 95% for both at approximately 25.5 seconds per question. "
        "Calibrated routing improves accuracy substantially while avoiding many strong-model calls. For example, the "
        "difficulty strategy at the budget operating point reaches 85% accuracy at 12.30 seconds for pair 1 and 77% "
        "at 12.43 seconds for pair 2. Self-consistency is stronger when high recall is required, reaching 91% and 92% "
        "accuracy, respectively, but its sampled calls add measurable overhead. The findings demonstrate that routing "
        "signals are useful, but their performance depends on calibration, model pairing, and the operational cost "
        "assumption."),
    P("<b>Keywords:</b> large language models, model routing, escalation, self-consistency, GSM8K, inference cost, Ollama"),
    PageBreak(),
]

story += [
    section("TABLE OF CONTENTS"),
    P("1. Introduction", "TOC"),
    P("2. Problem Statement and Objectives", "TOC"),
    P("3. Background and Related Work", "TOC"),
    P("4. System Design and Methodology", "TOC"),
    P("5. Implementation", "TOC"),
    P("6. Experimental Results", "TOC"),
    P("7. Discussion", "TOC"),
    P("8. Limitations", "TOC"),
    P("9. Conclusion and Future Scope", "TOC"),
    P("References", "TOC"),
    PageBreak(),
]

story += [
    section("1. INTRODUCTION"),
    P(
        "The use of language models in real systems creates a practical tradeoff between answer quality and inference "
        "cost. A small model is generally faster and cheaper but may fail on difficult or unusual inputs. A larger model "
        "is more capable, but invoking it for every request increases latency and resource consumption. Model routing "
        "addresses this tradeoff by predicting when the small model is likely to fail and escalating only those cases."),
    P(
        "This project is an experimental framework for evaluating that idea under controlled conditions. It does not "
        "present a new end-user chatbot. Instead, it collects model outputs once, stores their measured compute times, "
        "and replays multiple routing policies offline. This separation makes comparisons reproducible and ensures that "
        "each policy is evaluated on the same questions and recorded model behavior."),
    subsection("Research question"),
    P(
        "Can inexpensive signals identify questions that should be escalated to a stronger language model, thereby "
        "improving accuracy at lower average compute cost than always using the strong model?"),
    subsection("Contributions"),
    P(
        "The project contributes: (i) a leakage-checked pilot, calibration, and evaluation protocol; (ii) a local "
        "Ollama-based measurement pipeline; (iii) offline replay of several routing strategies under two cost "
        "conventions; (iv) cost-matched and transfer comparisons; and (v) plots and CSV summaries that support "
        "analysis beyond a single accuracy number."),
]

story += [
    section("2. PROBLEM STATEMENT AND OBJECTIVES"),
    P(
        "Given a question q, a small model S, and a strong model T, the router must decide whether to return S(q) or "
        "pay the additional cost of T(q). The router cannot use the ground-truth answer at deployment time. A useful "
        "router should therefore increase final accuracy while keeping escalation rate, latency, and signal overhead "
        "under control."),
    subsection("Objectives"),
    P("1. Measure the accuracy and compute time of candidate local models on GSM8K questions."),
    P("2. Compare routing signals based on randomness, question difficulty, answer agreement, and token confidence."),
    P("3. Calibrate operating points using a separate calibration split rather than evaluation labels."),
    P("4. Evaluate two-tier and three-tier cascades, including a medium model."),
    P("5. Compare the local signals with a RouteLLM BERT router."),
    P("6. Report quality-cost tradeoffs, confidence intervals, transfer behavior, and routing overhead."),
]

story += [
    section("3. BACKGROUND AND RELATED WORK"),
    subsection("Large language model inference"),
    P(
        "Autoregressive generation incurs computational cost that grows with model size and generated sequence length. "
        "For workloads containing many easy questions, sending every request to the largest available model is often "
        "unnecessary. A cascade can use a smaller model as a first attempt and reserve a stronger model for uncertain "
        "cases."),
    subsection("Self-consistency"),
    P(
        "Self-consistency samples multiple reasoning paths and uses answer agreement as a confidence signal. In this "
        "project, the small model produces k = 3 sampled answers. Low majority agreement triggers escalation; high "
        "agreement returns the majority answer. The sampled calls are explicitly counted as signal overhead."),
    subsection("Learned routing"),
    P(
        "RouteLLM-style routers learn to select between models using a preference or quality signal. This project uses "
        "the local BERT router configuration because it can run without OpenAI embeddings. Its scores are computed in "
        "a separate environment and then consumed by the same replay framework."),
    subsection("Position of this work"),
    P(
        "The emphasis here is not only the routing decision but also its measurement protocol. The project compares "
        "signals at matched escalation rates, separates calibration from evaluation, reports the cost of the signal "
        "itself, and runs a strict sensitivity analysis in which the initial small-model call is always paid."),
]

story += [
    section("4. SYSTEM DESIGN AND METHODOLOGY"),
    subsection("Dataset and splits"),
    P(
        "The project uses the OpenAI GSM8K dataset, which contains grade-school mathematical word problems. The data "
        "preparation script creates three non-overlapping splits: a 25-question pilot set from training data, a "
        "50-question calibration set from disjoint training data, and a 100-question evaluation set from test data. "
        "The script checks both question identifiers and question text for overlap."),
    table(
        [
            ["Split", "Source", "Size", "Purpose"],
            ["Pilot", "Train", "25", "Select and sanity-check model pairs"],
            ["Calibration", "Train", "50", "Fit difficulty statistics and thresholds"],
            ["Evaluation", "Test", "100", "Final comparison of routing strategies"],
        ],
        [3.0 * cm, 3.0 * cm, 2.0 * cm, 9.0 * cm],
    ),
    Spacer(1, 0.25 * cm),
    subsection("Models and generation"),
    P(
        "The primary configuration evaluates Qwen 2.5 1.5B and Llama 3.2 1B as small models, Qwen 2.5 7B as the "
        "strong model, and Qwen 2.5 3B as the medium tier. Models are served through Ollama at localhost. Greedy "
        "generation uses temperature 0.0, sampled generation uses temperature 0.7, the context limit is 4096 tokens, "
        "and generation is capped at 512 predicted tokens. Each answer is required to end with a numeric "
        "“Final answer” line, which is extracted and compared with the reference answer."),
    subsection("Routing signals"),
    table(
        [
            ["Signal", "Decision rule", "Runtime information"],
            ["Random", "Escalate with fixed probability p", "Question id and seed"],
            ["Difficulty", "Escalate when standardized text features are high", "Word, number, sentence counts"],
            ["E1 agreement", "Escalate when sampled-answer agreement is below threshold", "Three small-model samples"],
            ["E2 log probability", "Escalate when mean/min token probability is low", "Greedy-call log probabilities"],
            ["Profile", "Escalate in buckets where small accuracy is below target", "Question difficulty bucket"],
            ["RouteLLM", "Escalate when BERT score exceeds threshold", "Precomputed local router score"],
        ],
        [3.1 * cm, 8.0 * cm, 5.0 * cm],
    ),
    Spacer(1, 0.25 * cm),
    subsection("Calibration and operating points"),
    P(
        "Thresholds are selected on calibration records only. The budget operating point escalates at most 30% of "
        "calibration questions and maximizes captured small-model errors. The recall operating point selects the lowest "
        "escalation rate that catches at least 70% of small-model errors. Evaluation labels are used only after the "
        "routing rule has been fixed. Random and matched-rate baselines use the signal’s escalation rate but do not use "
        "labels to make decisions."),
    subsection("Cost conventions"),
    P(
        "In the primary pre-generation convention, a router that decides to escalate before generation skips the small "
        "call. In the strict convention, every question first pays for the small call, even when escalation follows. "
        "This sensitivity analysis prevents the results from depending on an optimistic deployment assumption."),
]

story += [
    section("5. IMPLEMENTATION"),
    P(
        "The repository is organized into configuration, data, model adapters, routing strategies, experiments, "
        "metrics, scripts, tests, results, and figures. The central YAML file keeps experiment settings out of the "
        "implementation. Model calls are collected as JSONL records containing the question id, extracted answer, "
        "correctness, generation statistics, and optional samples or log probabilities."),
    table(
        [
            ["Component", "Role"],
            ["src/config.py", "Load YAML configuration and resolve repository paths"],
            ["src/models/ollama_adapter.py", "Call local Ollama models and record timing"],
            ["src/experiments/collect_outputs.py", "Collect resumable JSONL model outputs"],
            ["src/routing/strategies.py", "Pure-function routing signals and outcomes"],
            ["src/experiments/replay.py", "Calibrate thresholds and replay strategies offline"],
            ["src/routing/tiered.py", "Capability profile, two-stage, and three-tier cascades"],
            ["src/routing/routellm_strategy.py", "Apply precomputed RouteLLM scores"],
            ["src/experiments/plots.py", "Generate quality-cost and transfer figures"],
            ["tests/", "Unit and integration tests for adapters, routing, replay, and plots"],
        ],
        [6.0 * cm, 10.1 * cm],
    ),
    Spacer(1, 0.25 * cm),
    subsection("Reproducibility"),
    P(
        "The experiment seed is 42. Random baselines average over 200 deterministic seeds, bootstrap confidence "
        "intervals use 10,000 resamples, and cached model outputs make replay repeatable. The two primary execution "
        "commands are <font name='Courier'>python scripts\\run_replay.py</font> and "
        "<font name='Courier'>python scripts\\run_tiered_replay.py</font>. The optional RouteLLM path uses "
        "<font name='Courier'>python scripts\\run_routellm_replay.py</font> after score generation."),
]

story += [
    section("6. EXPERIMENTAL RESULTS"),
    subsection("Baseline comparison"),
    P(
        "The two model pairs show the expected quality-cost gap. Pair 1 uses Qwen 2.5 1.5B with Qwen 2.5 7B; pair 2 "
        "uses Llama 3.2 1B with the same strong model. The strong-only baseline achieves 95% accuracy for both pairs "
        "at 25.48 seconds per question. The small-only baselines are much faster but achieve 71% and 54%. The oracle "
        "row is an analysis ceiling that uses labels to escalate only when the small answer is wrong and is not a "
        "deployable method."),
    table(
        [
            ["Pair", "Strategy", "Accuracy", "Cost s/q", "Escalation"],
            ["1", "Small-only", "71%", "2.48", "0%"],
            ["1", "Strong-only", "95%", "25.48", "100%"],
            ["1", "E1, budget", "88%", "14.08", "24%"],
            ["1", "Difficulty, budget", "85%", "12.30", "35%"],
            ["2", "Small-only", "54%", "2.75", "0%"],
            ["2", "Strong-only", "95%", "25.48", "100%"],
            ["2", "E1, budget", "84%", "17.96", "36%"],
            ["2", "Difficulty, budget", "77%", "12.43", "35%"],
        ],
        [2.0 * cm, 5.0 * cm, 2.5 * cm, 3.0 * cm, 3.0 * cm],
    ),
    Spacer(1, 0.25 * cm),
    figure("fig1_quality_vs_cost.png", "Figure 1. Accuracy versus mean compute cost for the routing curves and calibrated points."),
    PageBreak(),
    subsection("Self-consistency and recall"),
    P(
        "Self-consistency is effective when the objective is to catch a large fraction of small-model errors. At the "
        "recall operating point, E1 reaches 91% accuracy for pair 1 and 92% for pair 2. However, the three sampled "
        "small-model calls create 6.55 to 6.96 seconds of overhead per question on average, and high recall causes "
        "many questions to be escalated. The budget point is cheaper, reaching 88% and 84% accuracy at 24% and 36% "
        "escalation, respectively."),
    figure("fig2_accuracy_vs_escalation.png", "Figure 2. Accuracy as a function of escalation rate."),
    subsection("Three-tier cascade"),
    P(
        "The three-tier experiment adds Qwen 2.5 3B between the small and strong models. For pair 1, the three-tier "
        "configuration with thresholds t1 = 0.5 and t2 = 0.5 reaches 87% accuracy at 12.66 seconds per question, "
        "while t1 = 0.833 and t2 = 0.5 reaches 90% at 17.04 seconds. The highest tested setting, t1 = 0.833 and "
        "t2 = 0.833, reaches 92% at 22.03 seconds. This demonstrates that a medium tier can improve the quality-cost "
        "frontier, but it also introduces additional sampling overhead and calibration choices."),
    table(
        [
            ["Configuration", "Accuracy", "Cost s/q", "Small/medium/strong share"],
            ["t1=0.5, t2=0.5", "87%", "12.66", "76% / 20% / 4%"],
            ["t1=0.5, t2=0.833", "89%", "15.34", "76% / 12% / 12%"],
            ["t1=0.833, t2=0.5", "90%", "17.04", "48% / 48% / 4%"],
            ["t1=0.833, t2=0.833", "92%", "22.03", "48% / 32% / 20%"],
        ],
        [5.0 * cm, 3.0 * cm, 3.0 * cm, 6.0 * cm],
    ),
    Spacer(1, 0.25 * cm),
    figure("fig3_latency.png", "Figure 3. Mean latency, separating model-call time from routing signal overhead."),
    PageBreak(),
    subsection("RouteLLM comparison"),
    P(
        "The local RouteLLM BERT router provides an additional learned baseline. At its calibrated budget operating "
        "point, it reaches 84% accuracy at 12.06 seconds for pair 1 and 73% at 10.98 seconds for pair 2. The result "
        "is competitive with simple signals in some settings but is not uniformly better. In particular, pair 2 shows "
        "that a router trained or calibrated for one model relationship may transfer imperfectly to another."),
    subsection("Transfer and matched-cost analysis"),
    P(
        "The cost-matched tables compare each calibrated signal with random and difficulty curves at the same measured "
        "cost. For pair 1, E1 at the budget point achieves 88%, compared with 77% for a matched random policy and 80% "
        "for the difficulty curve. For pair 2, E1 at the budget point achieves 84%, compared with 69% random and 78% "
        "difficulty. These comparisons indicate that agreement contains useful information beyond escalation rate "
        "alone, although the benefit is smaller at some recall settings."),
    figure("fig4_transfer.png", "Figure 4. Transfer behavior when thresholds or operating points move between model pairs."),
]

story += [
    section("7. DISCUSSION"),
    subsection("Accuracy-cost frontier"),
    P(
        "No single router dominates across all operating points. Small-only routing is the cheapest option but leaves "
        "many errors uncorrected. Strong-only routing gives a stable high-quality reference but spends roughly ten times "
        "the pair 1 small-only cost. Calibrated difficulty routing provides a low-overhead middle point, while E1 "
        "agreement can provide higher accuracy when the system can afford sampled calls."),
    subsection("The value of calibration"),
    P(
        "Calibration is essential because the same threshold does not necessarily produce the same escalation rate or "
        "accuracy for different small models. The pair 1 and pair 2 results differ even though they use the same strong "
        "model. The transfer analysis therefore treats thresholds as model-pair-specific unless evidence supports "
        "reuse."),
    subsection("Operational interpretation"),
    P(
        "The primary pre-generation convention represents a router capable of deciding before the small model runs. "
        "The strict results are more conservative and should be preferred when the router depends on a small-model "
        "signal that is unavailable before generation. In production, the correct convention depends on the actual "
        "signal pipeline and must be measured rather than assumed."),
]

story += [
    section("8. LIMITATIONS"),
    P("1. The evaluation set contains 100 questions, so individual percentage points should not be overinterpreted."),
    P("2. The study uses a single task domain, GSM8K mathematics, and may not transfer to language, coding, or multimodal tasks."),
    P("3. Local generation times depend on hardware, Ollama configuration, model quantization, and system load."),
    P("4. The evaluation measures recorded completion time rather than cloud billing cost or energy consumption."),
    P("5. Difficulty features are intentionally simple and may not capture semantic difficulty."),
    P("6. Self-consistency uses only three samples; more samples could improve confidence but increase overhead."),
    P("7. The RouteLLM baseline is limited to the local BERT configuration and precomputed scores available in this repository."),
    P("8. Calibration and evaluation are separated, but the small sample sizes can still produce noisy threshold estimates."),
]

story += [
    section("9. CONCLUSION AND FUTURE SCOPE"),
    P(
        "This project demonstrates an end-to-end, reproducible methodology for evaluating cost-aware LLM routing. "
        "On GSM8K, routing can recover a substantial portion of the strong model’s accuracy without always invoking "
        "the strong model. Difficulty-based routing is attractive when low overhead is important, while self-consistency "
        "is more effective when the target is high error recall. A medium tier and a learned router provide useful "
        "additional points on the quality-cost frontier, but their benefits depend on calibration and model pairing."),
    subsection("Future scope"),
    P("1. Expand evaluation to larger and more diverse datasets, including coding and natural-language tasks."),
    P("2. Train a task-specific router using richer semantic features and calibrated uncertainty estimates."),
    P("3. Evaluate adaptive sample counts instead of fixing self-consistency at k = 3."),
    P("4. Incorporate monetary cost, GPU utilization, energy, batching, and concurrent serving effects."),
    P("5. Add statistical significance tests across repeated dataset samples and hardware configurations."),
    P("6. Investigate online calibration and drift detection as model versions or workloads change."),
    PageBreak(),
    section("REFERENCES"),
    P("[1] Cobbe, K. et al. “Training Verifiers to Solve Math Word Problems.” arXiv, 2021. GSM8K dataset."),
    P("[2] Wang, X. et al. “Self-Consistency Improves Chain of Thought Reasoning in Language Models.” ICLR, 2023."),
    P("[3] Ong, I. et al. “RouteLLM: Learning to Route LLMs with Preference Data.” arXiv, 2024."),
    P("[4] Ollama. “Get up and running with large language models locally.” https://ollama.com/"),
    P("[5] Hugging Face Datasets. “GSM8K dataset documentation.” https://huggingface.co/datasets/openai/gsm8k"),
    P("[6] NumPy, pandas, SciPy, Matplotlib, PyYAML, requests, and pytest project documentation."),
    Spacer(1, 0.5 * cm),
    P("<b>Appendix A: Main project commands</b>", "Heading2Report"),
    P("python scripts\\prepare_data.py", "SmallReport"),
    P("python scripts\\run_pilot.py", "SmallReport"),
    P("python scripts\\run_collect.py --model qwen2.5:1.5b --split evaluation --role small", "SmallReport"),
    P("python scripts\\run_replay.py", "SmallReport"),
    P("python scripts\\run_tiered_replay.py", "SmallReport"),
    P("python scripts\\run_plots.py", "SmallReport"),
]


OUT.parent.mkdir(parents=True, exist_ok=True)
doc = NumberedDocTemplate(
    str(OUT),
    pagesize=A4,
    rightMargin=1.7 * cm,
    leftMargin=1.7 * cm,
    topMargin=1.6 * cm,
    bottomMargin=1.7 * cm,
    title="Cost-Aware Routing of Large Language Models",
    author="[STUDENT NAME]",
)
doc.build(story)
print(OUT)

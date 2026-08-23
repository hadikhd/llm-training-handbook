id: outlook
title: چشم‌انداز
sidebar_position: 11
sidebar_label: چشم‌انداز
---
## چشم‌انداز

<div className="chapter-hero">

![فصل ۱۱:  چشم‌انداز]([]/img/chapters/outlook.png)

</div>

[قبلی: سامانه‌ها (Systems)](./10-inference-and-decoding.md) |
[فهرست مطالب (Contents)](./index.md) |

---


## اهداف یادگیری (Learning Objectives)

تا پایان این فصل، باید بتوانید:

- اصول پایدار مهندسی (Durable Engineering Principles) را از روندهای سریع‌التغییر مدل‌های زبانی بزرگ (LLM Trends) جدا کنید.
- درک کنید چرا مقیاس‌پذیری کارآمد (Efficient Scaling)، کیفیت داده (Data Quality)، ارزیابی (Evaluation)، و سامانه‌ها (Systems) همچنان محوری باقی می‌مانند.
- جهت‌گیری‌های نوظهور مانند استدلال (Reasoning)، عامل‌ها (Agents)، چندوجهی‌بودن (Multimodality)، زمینهٔ بلند (Long Context)، و مدل‌های تخصصی (Specialized Models) را شناسایی کنید.
- یک نقشه‌راه یادگیری عملی (Practical Learning Roadmap) برای ادامهٔ کار مهندسی مدل‌های زبانی بزرگ (LLM Engineering) بسازید.

---

## لنز مهندسی (Engineering Lens)

این فصل را با چهار پرسش مهندسی (Engineering Questions) در ذهن بخوانید:

1. **این تکنیک (Technique) چه مسئله‌ای را حل می‌کند؟**
2. **چگونه اندازه‌گیری می‌کنیم که آیا کار می‌کند یا نه؟**
3. **چه موازنه‌ها (Trade-offs) یا هزینه‌های منابعی (Resource Costs) ایجاد می‌کند؟**
4. **چه حالت‌های شکست (Failure Modes) را باید انتظار داشته باشیم، و چگونه آن‌ها را تشخیص می‌دهیم؟**

## ۱. چرا چشم‌انداز آینده مهم است (Why Outlook Matters)

مدل‌های زبانی بزرگ (Large Language Models) یک فناوری تمام‌شده نیستند. آن‌ها یک مرز فعال مهندسی (Active Engineering Frontier) هستند.

دستورالعمل اصلی (Core Recipe) اکنون آشناست:

```text
large-scale data
+ Transformer-based architecture
+ next-token prediction
+ post-training
+ evaluation
+ inference system
```
اما تقریباً هر بخش از این دستورالعمل در حال تغییر است.

پیشرفت آینده فقط از بزرگ‌تر کردن مدل‌ها حاصل نخواهد شد. بلکه از دادهٔ بهتر (Better Data)، معماری‌های بهتر (Better Architectures)، اهداف آموزشی بهتر (Better Training Objectives)، ارزیابی بهتر (Better Evaluation)، سامانه‌های استنتاج بهتر (Better Inference Systems)، استفادهٔ قوی‌تر از ابزار (Stronger Tool Use)، استدلال چندوجهی (Multimodal Reasoning)، و شیوه‌های استقرار قابل‌اعتمادتر (More Reliable Deployment Practices) نیز حاصل خواهد شد.

پرسش مرکزی دیگر این نیست:

```text
Can we train a large language model?
```
بلکه به‌طور فزاینده این است:

```text
Can we build models that are useful, reliable, efficient, controllable, and economically sustainable?
```
---

## ۲. از مقیاس‌پذیری تا مقیاس‌پذیری کارآمد (From Scaling to Efficient Scaling)

موج نخست پیشرفت مدل‌های زبانی بزرگ مدرن (Modern LLM Progress) به‌شدت توسط مقیاس (Scale) هدایت شد.

مدل‌های بزرگ‌تر (Larger Models)، مجموعه‌داده‌های بزرگ‌تر (Larger Datasets)، و بودجه‌های محاسباتی بزرگ‌تر (Larger Compute Budgets) بهبودهای قوی ایجاد کردند. قوانین مقیاس‌پذیری (Scaling Laws) به مهندسان کمک کردند رابطهٔ میان پارامترها (Parameters)، توکن‌ها (Tokens)، محاسبه (Compute)، و زیان (Loss) را برآورد کنند.

بااین‌حال، مقیاس‌پذیری ساده‌انگارانه (Naive Scaling) محدودیت‌هایی دارد:

- محاسبه (Compute) گران است
- دادهٔ باکیفیت (High-Quality Data) محدود است
- هزینهٔ استنتاج (Inference Cost) اهمیت دارد
- تأخیر (Latency) اهمیت دارد
- محدودیت‌های انرژی و سخت‌افزار (Energy and Hardware Constraints) اهمیت دارند
- استقرار مدل‌های بزرگ‌تر (Larger Models) دشوارتر است
- ارزیابی (Evaluation) در سطوح قابلیت بالاتر دشوارتر می‌شود

مرحلهٔ بعدی احتمالاً بر مقیاس‌پذیری کارآمد (Efficient Scaling) تمرکز خواهد کرد.

مقیاس‌پذیری کارآمد (Efficient Scaling) می‌پرسد:

```text
How much capability can be gained per unit of data, compute, memory, and cost?
```
این موضوع هدف بهینه‌سازی (Optimization Target) را از بیشینه‌سازی اندازهٔ مدل (Maximum Model Size) به بیشینه‌سازی قابلیت مفید (Maximum Useful Capability) تحت محدودیت‌های واقعی (Real Constraints) تغییر می‌دهد.

---

## ۳. کیفیت داده مهم‌تر خواهد شد (Data Quality Will Become More Important)

داده (Data) صرفاً سوخت آموزش (Fuel for Training) نیست. داده شکل می‌دهد که مدل چه می‌داند، چگونه استدلال می‌کند، از چه زبان‌هایی پشتیبانی می‌کند، چه قالب‌هایی را دنبال می‌کند، و چه حالت‌های شکستی (Failure Modes) در آن شکل می‌گیرد.

توسعهٔ آیندهٔ مدل‌های زبانی بزرگ (Future LLM Development) احتمالاً تأکید بیشتری بر موارد زیر خواهد داشت:

- پالایش بهتر اسناد (Better Document Filtering)
- حذف تکرار قوی‌تر (Stronger Deduplication)
- اعتبارسنجی دادهٔ مصنوعی (Synthetic-Data Validation)
- طراحی برنامهٔ درسی (Curriculum Design)
- ترکیب‌های دادهٔ دامنه‌محور (Domain-Specific Data Mixtures)
- پوشش زبان‌های چندزبانه و کم‌منبع (Multilingual and Low-Resource Language Coverage)
- ردیابی منشأ داده (Provenance Tracking)
- مدیریت داده با آگاهی از مجوز (License-Aware Data Management)
- تشخیص آلودگی (Contamination Detection)
- دادهٔ ترجیح باکیفیت‌تر (Higher-Quality Preference Data)
- مثال‌های بهتر برای زمینهٔ بلند (Better Long-Context Examples)
- مسیرهای استفاده از ابزار (Tool-Use Trajectories)
- ردپاهای استدلال (Reasoning Traces) در موارد مناسب

با افزایش ظرفیت مدل (Model Capacity)، دادهٔ کم‌کیفیت (Low-Quality Data) می‌تواند به گلوگاه بزرگ‌تری (Larger Bottleneck) تبدیل شود. یک مدل قوی‌تر ممکن است هم الگوهای مفید (Useful Patterns) و هم مصنوعات نامطلوب (Undesirable Artifacts) را مؤثرتر یاد بگیرد.

درس عملی این است:

```text
Better data engineering often beats simply adding more data.
```
---

## ۴. دادهٔ مصنوعی و خودبهبودی (Synthetic Data and Self-Improvement)

دادهٔ مصنوعی (Synthetic Data) در توسعهٔ مدل‌های زبانی بزرگ (LLM Development) اهمیت فزاینده‌ای دارد.

می‌توان از آن برای موارد زیر استفاده کرد:

- تنظیم با دستورالعمل (Instruction Tuning)
- مثال‌های استدلال (Reasoning Examples)
- نمایش‌های استفاده از ابزار (Tool-Use Demonstrations)
- وظایف تولید کد (Code Generation Tasks)
- پرسش‌وپاسخ دامنه‌محور (Domain-Specific Question Answering)
- ساخت مجموعهٔ ارزیابی (Evaluation Set Construction)
- مقایسهٔ ترجیح (Preference Comparison)
- گسترش چندزبانه (Multilingual Expansion)
- آموزش ایمنی (Safety Training)
- مسیرهای عامل (Agent Trajectories)

بااین‌حال، دادهٔ مصنوعی (Synthetic Data) ریسک‌هایی ایجاد می‌کند:

- تقویت خطا (Error Amplification)
- کاهش تنوع (Reduced Diversity)
- فروپاشی سبک (Style Collapse)
- سوگیری‌های پنهان مدل (Hidden Model Biases)
- نادرستی‌های واقعی (Factual Inaccuracies)
- آلودگی بنچمارک (Benchmark Contamination)
- بیش‌برازش به الگوهای مولد (Overfitting to Generator Patterns)
- اعتماد کاذب به برچسب‌های تولیدشده (False Confidence in Generated Labels)

دادهٔ مصنوعی (Synthetic Data) زمانی بیشترین فایده را دارد که پالایش (Filtered)، اعتبارسنجی (Validated)، متنوع‌سازی (Diversified)، و با دادهٔ انسانی یا واقعی باکیفیت (High-Quality Human or Real-World Data) ترکیب شود.

یک جریان کاری عملی دادهٔ مصنوعی (Practical Synthetic-Data Workflow) چنین است:

```text
generate candidates
-> filter for quality and diversity
-> verify where possible
-> mix with trusted data
-> train or fine-tune
-> evaluate on independent tests
```
خودبهبودی (Self-Improvement) امیدبخش است، اما خودآموزی کنترل‌نشده (Uncontrolled Self-Training) می‌تواند مدل را بدون درست‌تر کردن، فقط مطمئن‌تر کند.

---

## ۵. مدل‌های استدلالی (Reasoning Models)

پیشرفت‌های اخیر باعث افزایش علاقه به مدل‌هایی شده است که در زمان استنتاج (Inference Time) محاسبهٔ بیشتری مصرف می‌کنند تا وظایف دشوار را حل کنند.

به‌جای تولید فوری پاسخ، مدل ممکن است استدلال میانی (Intermediate Reasoning) تولید کند، میان راه‌حل‌های ممکن جست‌وجو کند (Search over Possible Solutions)، ابزارها را فراخوانی کند (Call Tools)، مراحل را راستی‌آزمایی کند (Verify Steps)، یا پاسخ خود را بازبینی کند (Revise its Answer).

این موضوع الگوی محاسباتی (Compute Pattern) را تغییر می‌دهد:

standard inference:
one forward generation path

```text
reasoning-oriented inference:
generate
check
branch
verify
refine
```
مدل‌های استدلالی (Reasoning Models) برای موارد زیر مفید هستند:

```text
- mathematics
- programming
- planning
- scientific problem solving
- multi-step analysis
- symbolic tasks
- agentic workflows
```
اما چالش‌های جدیدی ایجاد می‌کنند:

- هزینهٔ استنتاج بالاتر (Higher Inference Cost)
- تأخیر طولانی‌تر (Longer Latency)
- ارزیابی دشوارتر (Harder Evaluation)
- شکست‌های پنهان استدلال (Hidden Reasoning Failures)
- کنترل پرگویی (Verbosity Control)
- هک پاداش (Reward Hacking)
- خودراستی‌آزمایی غیرقابل‌اعتماد (Unreliable Self-Verification)
- دشواری در تمایز استدلال معتبر از متن محتمل (Difficulty Distinguishing Valid Reasoning from Plausible Text)

چالش کلیدی فقط واداشتن مدل‌ها به طولانی‌تر فکر کردن نیست. بلکه این است که محاسبهٔ اضافی (Additional Computation) پاسخ‌های قابل‌اعتمادتر (More Reliable Answers) تولید کند.

---

## ۶. محاسبه در زمان آزمون (Test-Time Compute)

محاسبه در زمان آزمون (Test-Time Compute) به استفاده از محاسبهٔ بیشتر در طول استنتاج (Inference) برای بهبود کیفیت خروجی (Output Quality) اشاره دارد.

نمونه‌ها شامل موارد زیر هستند:

- نمونه‌گیری چندین پاسخ (Sampling Multiple Answers)
- رأی‌گیری اکثریت (Majority Voting)
- خودسازگاری (Self-Consistency)
- جست‌وجو روی مسیرهای استدلال (Search over Reasoning Paths)
- تولید هدایت‌شده با راستی‌آزما (Verifier-Guided Generation)
- حل مسئله با کمک ابزار (Tool-Assisted Solving)
- حلقه‌های نقد و بازبینی (Critique-and-Revise Loops)
- اجرای برنامه (Program Execution)
- گسترش بازیابی (Retrieval Expansion)

موازنهٔ پایه (Basic Trade-off) این است:

```text
more inference compute
in exchange for
potentially better answers
```
این روش زمانی مفید است که درستی (Correctness) از تأخیر (Latency) یا هزینه (Cost) مهم‌تر باشد.

نمونه‌ها:

| وظیفه (Task) | محاسبهٔ مفید در زمان آزمون (Useful Test-Time Compute) |
|---|---|
| حل مسئلهٔ ریاضی (Math Problem Solving) | چند مسیر حل و راستی‌آزمایی (Multiple Solution Paths and Verification) |
| تولید کد (Code Generation) | تولید، اجرای آزمون‌ها، ترمیم (Generate, Run Tests, Repair) |
| RAG | بازیابی، پاسخ، بررسی ارجاعات (Retrieve, Answer, Check Citations) |
| برنامه‌ریزی (Planning) | شبیه‌سازی گزینه‌های جایگزین (Simulate Alternatives) |
| تحلیل داده (Data Analysis) | اجرای ابزارها و اعتبارسنجی خروجی‌ها (Execute Tools and Validate Outputs) |

محاسبه در زمان آزمون (Test-Time Compute) باید به‌عنوان یک بودجهٔ مهندسی صریح (Explicit Engineering Budget) در نظر گرفته شود. تولید پاسخ‌های بیشتر به‌صورت خودکار نتایج بهتری ایجاد نمی‌کند، مگر این‌که سامانه بتواند بهترین خروجی را انتخاب یا راستی‌آزمایی کند.

---

## ۷. راستی‌آزماها و مدل‌های پاداش (Verifiers and Reward Models)

یک مولد (Generator) پاسخ‌های نامزد (Candidate Answers) تولید می‌کند. یک راستی‌آزما (Verifier) آن‌ها را ارزیابی می‌کند.

مدل‌های راستی‌آزما (Verifier Models) می‌توانند در موارد زیر کمک کنند:

- رتبه‌بندی خروجی‌های نامزد (Ranking Candidate Outputs)
- بررسی مراحل ریاضی (Checking Mathematical Steps)
- اعتبارسنجی رفتار کد (Validating Code Behavior)
- تشخیص ادعاهای بدون پشتیبانی (Detecting Unsupported Claims)
- امتیازدهی به نتایج فراخوانی ابزار (Scoring Tool-Call Results)
- انتخاب میان مسیرهای استدلال (Selecting Among Reasoning Paths)
- بهبود بهینه‌سازی ترجیح (Improving Preference Optimization)

یک الگوی ساده (Simple Pattern) چنین است:

```text
generate several candidate answers
-> score each answer with a verifier
-> select or refine the best candidate
```
راستی‌آزماهای خوب (Good Verifiers) می‌توانند قابلیت اعتماد (Reliability) را بهبود دهند. راستی‌آزماهای ضعیف (Poor Verifiers) می‌توانند حس کاذبی از درستی (False Sense of Correctness) ایجاد کنند.

طراحی راستی‌آزما (Verifier Design) نیازمند توجه به موارد زیر است:

- کالیبراسیون (Calibration)
- استحکام (Robustness)
- اختصاصی‌بودن نسبت به وظیفه (Task Specificity)
- مثال‌های خصمانه (Adversarial Examples)
- جابه‌جایی توزیع (Distribution Shift)
- توضیح‌پذیری (Explainability)
- استقلال از مولد (Independence from the Generator)
- ارزیابی در برابر حقیقت مبنا (Evaluation against Ground Truth)

با توانمندتر شدن مدل‌ها، داوری خروجی‌ها (Judging Outputs) ممکن است به‌اندازهٔ تولید آن‌ها مهم شود.

---

## ۸. سامانه‌های ابزارمحور و عاملی (Tool-Using and Agentic Systems)

مدل‌های زبانی بزرگ (LLMs) به‌طور فزاینده به‌عنوان مؤلفه‌هایی درون سامانه‌های بزرگ‌تر استفاده می‌شوند که می‌توانند اطلاعات را بازیابی کنند، APIها را فراخوانی کنند، کد بنویسند، فایل‌ها را دست‌کاری کنند، جریان‌های کاری را کنترل کنند، و با محیط‌های خارجی تعامل داشته باشند.

یک حلقهٔ عاملی پایه (Basic Agentic Loop) چنین است:

```text
observe
-> decide
-> act
-> receive feedback
-> update context
-> continue or stop
```
سامانه‌های ابزارمحور (Tool-Using Systems) می‌توانند محدودیت‌های مدل مستقل (Standalone Model) را با افزودن موارد زیر پشت سر بگذارند:

- جست‌وجو (Search)
- ماشین‌حساب‌ها (Calculators)
- پایگاه‌های داده (Databases)
- اجرای کد (Code Execution)
- APIهای دامنه‌ای (Domain APIs)
- سامانه‌های حافظه (Memory Systems)
- شبیه‌سازها (Simulators)
- ابزارهای راستی‌آزمایی (Verification Tools)
- جریان‌های کاری ساخت‌یافته (Structured Workflows)

اما استفاده از ابزار (Tool Use) ریسک را نیز افزایش می‌دهد:

- انتخاب نادرست ابزار (Incorrect Tool Selection)
- آرگومان‌های نامعتبر (Invalid Arguments)
- اقدامات ناایمن (Unsafe Actions)
- تزریق پرامپت (Prompt Injection)
- فراخوانی‌های بیش‌ازحد ابزار (Excessive Tool Calls)
- خرابی پنهان وضعیت (Hidden State Corruption)
- اشکال‌زدایی دشوار (Difficult Debugging)
- تأخیر و هزینهٔ بالاتر (Higher Latency and Cost)

جهت عملی (Practical Direction) احتمالاً به نفع عامل‌های محدود، مشاهده‌پذیر و مجوزدار (Constrained, Observable, Permissioned Agents) در برابر سامانه‌های خودمختار نامحدود (Unconstrained Autonomous Systems) خواهد بود.

عامل‌های قابل‌اعتماد (Reliable Agents) به موارد زیر نیاز دارند:

- شِماهای ابزار روشن (Clear Tool Schemas)
- مدیریت وضعیت (State Management)
- مرزهای مجوز (Permission Boundaries)
- اعتبارسنجی اقدام (Action Validation)
- راهبردهای بازگشت (Rollback Strategies)
- لاگ‌های ممیزی (Audit Logs)
- وظایف ارزیابی (Evaluation Tasks)
- بازیابی از شکست (Failure Recovery)

یک مدل زبانی بزرگ (LLM) باید به‌عنوان یک مؤلفهٔ تصمیم‌گیری (Decision Component) درون یک سامانهٔ کنترل‌شده (Controlled System) در نظر گرفته شود، نه به‌عنوان یک سامانهٔ کامل به‌تنهایی.

---

## ۹. مدل‌های تقویت‌شده با بازیابی (Retrieval-Augmented Models)

تولید تقویت‌شده با بازیابی (Retrieval-Augmented Generation) همچنان مهم خواهد ماند، زیرا وزن‌های مدل (Model Weights) به‌تنهایی نمی‌توانند هر مسئلهٔ دانشی (Knowledge Problem) را حل کنند.

RAG در موارد زیر کمک می‌کند:

- دانش تازه (Fresh Knowledge)
- پیکره‌های خصوصی (Private Corpora)
- پاسخ‌های قابل‌ردیابی (Traceable Answers)
- زمینه‌سازی دامنه‌محور (Domain-Specific Grounding)
- ریسک کمتر توهم (Lower Hallucination Risk)
- استقرار مدل کوچک‌تر (Smaller Model Deployment)
- پایگاه‌های دانش قابل‌پیکربندی (Configurable Knowledge Bases)

سامانه‌های RAG آینده احتمالاً پیچیده‌تر خواهند شد.

جهت‌ها شامل موارد زیر هستند:

- بازیابی ترکیبی تُنُک-چگال (Hybrid Sparse-Dense Retrieval)
- مسیریابی پرس‌وجوی آموخته‌شده (Learned Query Routing)
- بازیابی چندگامی (Multi-Hop Retrieval)
- بازیابی مبتنی بر گراف (Graph-Based Retrieval)
- بازرتبه‌بندی بهتر (Better Reranking)
- فشرده‌سازی زمینه (Context Compression)
- راستی‌آزمایی ارجاع (Citation Verification)
- تولید آگاه از بازیابی (Retrieval-Aware Generation)
- یکپارچه‌سازی دانش ساخت‌یافته (Structured Knowledge Integration)
- ادغام بازیابی با زمینهٔ بلند (Long-Context Retrieval Fusion)
- حافظه و شخصی‌سازی (Memory and Personalization)

یک سامانهٔ RAG قوی فقط یک پایگاه‌دادهٔ برداری (Vector Database) به‌علاوهٔ یک پرامپت (Prompt) نیست. بلکه یک خط لولهٔ کامل بازیابی، رتبه‌بندی، زمینه‌سازی، تولید، و ارزیابی (Retrieval, Ranking, Grounding, Generation, and Evaluation Pipeline) است.

---

## ۱۰. مدل‌های زمینهٔ بلند (Long-Context Models)

مدل‌های زمینهٔ بلند (Long-Context Models) می‌توانند ورودی‌های بزرگی مانند کتاب‌ها، کدبیس‌ها (Codebases)، اسناد حقوقی، مقالات پژوهشی، و تاریخچه‌های گفتگو را پردازش کنند.

آن‌ها امکان موارد زیر را فراهم می‌کنند:

- تحلیل در سطح سند (Document-Level Analysis)
- درک کد در سطح مخزن (Repository-Level Code Understanding)
- گفتگوی افق‌بلند (Long-Horizon Dialogue)
- سنتز چندسندی (Multi-Document Synthesis)
- ردپاهای گستردهٔ ابزار (Extended Tool Traces)
- پرامپت‌های RAG غنی‌تر (Richer RAG Prompts)

بااین‌حال، زمینهٔ بلند (Long Context) چالش‌هایی ایجاد می‌کند:

- هزینهٔ پیش‌پرکردن بالاتر (Higher Prefill Cost)
- کش KV بزرگ‌تر (Larger KV Cache)
- مشکلات کارایی توجه (Attention Efficiency Problems)
- استفادهٔ ضعیف‌تر از اطلاعات دور (Weaker Use of Distant Information)
- افت مرتبط با موقعیت (Position-Related Degradation)
- ارزیابی دشوارتر (Harder Evaluation)
- مسائل سازمان‌دهی پرامپت (Prompt Organization Issues)
- مواجههٔ بیشتر با تزریق پرامپت (Greater Exposure to Prompt Injection)

زمینهٔ بلند (Long Context) نیاز به بازیابی (Retrieval) یا خلاصه‌سازی (Summarization) را حذف نمی‌کند. بلکه نحوهٔ استفاده از آن‌ها را تغییر می‌دهد.

یک سامانهٔ عملی زمینهٔ بلند (Practical Long-Context System) همچنان به موارد زیر نیاز دارد:

- پالایش مرتبط‌بودن (Relevance Filtering)
- انتخاب بخش‌ها (Section Selection)
- پرامپت‌نویسی آگاه از سلسله‌مراتب (Hierarchy-Aware Prompting)
- ردیابی ارجاع (Citation Tracking)
- فشرده‌سازی زمینه (Context Compression)
- حل تعارض (Conflict Resolution)
- ارزیابی مقاوم (Robust Evaluation)

آینده احتمالاً زمینهٔ بلند (Long Context) را با بازیابی (Retrieval) ترکیب خواهد کرد، نه این‌که بازیابی را کاملاً جایگزین کند.

---

## ۱۱. سامانه‌های حافظه (Memory Systems)

حافظهٔ مدل زبانی بزرگ (LLM Memory) می‌تواند چند معنای متفاوت داشته باشد.

| نوع حافظه (Memory Type) | معنا (Meaning) |
|---|---|
| حافظهٔ پارامتری (Parametric Memory) | دانش ذخیره‌شده در وزن‌های مدل (Knowledge Stored in Model Weights) |
| حافظهٔ زمینه (Context Memory) | اطلاعاتی که در حال حاضر داخل پرامپت است (Information Currently Inside the Prompt) |
| حافظهٔ بازیابی (Retrieval Memory) | اسناد یا پایگاه‌های دادهٔ خارجی (External Documents or Databases) |
| حافظهٔ رویدادی (Episodic Memory) | تاریخچهٔ تعامل ذخیره‌شده (Stored Interaction History) |
| حافظهٔ کاری (Working Memory) | وضعیت موقت وظیفه (Temporary Task State) |
| حافظهٔ ابزار (Tool Memory) | وضعیت ذخیره‌شده توسط سامانه‌های خارجی (State Stored by External Systems) |

سامانه‌های حافظهٔ مفید (Useful Memory Systems) باید تصمیم بگیرند:

- چه چیزی را ذخیره کنند
- چه زمانی آن را ذخیره کنند
- چگونه آن را بازیابی کنند
- چگونه آن را به‌روزرسانی کنند
- چگونه آن را حذف کنند
- چگونه از آن محافظت کنند
- چگونه آن را ارزیابی کنند

حافظه (Memory) نگرانی‌های حریم خصوصی (Privacy)، درستی (Correctness)، و کنترل (Control) ایجاد می‌کند. ذخیره‌کردن همه‌چیز معمولاً طراحی ضعیفی است.

سامانه‌های حافظهٔ خوب (Good Memory Systems) گزینشی (Selective)، قابل‌بازرسی (Inspectable)، مجوزدار (Permissioned)، و برگشت‌پذیر (Reversible) هستند.

---

## ۱۲. مدل‌های چندوجهی (Multimodal Models)

مدل‌های زبانی بزرگ (LLMs) فراتر از متن (Text) در حال گسترش هستند.

سامانه‌های چندوجهی (Multimodal Systems) ممکن است موارد زیر را پردازش یا تولید کنند:

- تصاویر (Images)
- صوت (Audio)
- ویدئو (Video)
- اسناد (Documents)
- نمودارها (Charts)
- دیاگرام‌ها (Diagrams)
- کد (Code)
- دادهٔ حسگر (Sensor Data)
- رابط‌های کاربری (User Interfaces)
- جدول‌های ساخت‌یافته (Structured Tables)

یک مدل چندوجهی (Multimodal Model) ممکن است موارد زیر را ترکیب کند:

```text
vision encoder
+ audio encoder
+ language model
+ projection layers
+ multimodal training data
+ instruction tuning
```
قابلیت‌های رایج شامل موارد زیر هستند:

- کپشن‌گذاری تصویر (Image Captioning)
- پاسخ‌گویی به پرسش بصری (Visual Question Answering)
- درک اسناد شبیه OCR (OCR-like Document Understanding)
- تفسیر نمودار (Chart Interpretation)
- خلاصه‌سازی ویدئو (Video Summarization)
- تعامل گفتاری (Speech Interaction)
- جست‌وجوی چندوجهی (Multimodal Search)
- خودکارسازی رابط گرافیکی (GUI Automation)

سامانه‌های چندوجهی (Multimodal Systems) چالش‌های اضافی ایجاد می‌کنند:

- هم‌ترازی میان وجه‌ها (Alignment between Modalities)
- وضوح و بودجهٔ توکن (Resolution and Token Budget)
- استدلال زمانی (Temporal Reasoning)
- زمینه‌سازی بصری (Visual Grounding)
- توهم دربارهٔ تصاویر (Hallucination about Images)
- خطاهای OCR
- مسائل ایمنی در محتوای بصری (Safety Issues in Visual Content)
- پیچیدگی ارزیابی (Evaluation Complexity)

آیندهٔ مدل‌های زبانی بزرگ (LLMs) احتمالاً به‌طور فزاینده چندوجهی (Multimodal) خواهد بود، با زبان به‌عنوان رابط میان ادراک (Perception)، استدلال (Reasoning)، و اقدام (Action).

---

## ۱۳. مدل‌های کوچک‌تر تخصصی (Smaller Specialized Models)

پیشرفت به مدل‌های مقیاس مرزی (Frontier-Scale Models) محدود نیست.

مدل‌های کوچک‌تر (Smaller Models) مهم هستند، زیرا می‌توانند:

- ارزان‌تر برای سرویس‌دهی باشند (Cheaper to Serve)
- سریع‌تر باشند (Faster)
- آسان‌تر به‌صورت خصوصی مستقر شوند (Easier to Deploy Privately)
- آسان‌تر تنظیم دقیق شوند (Easier to Fine-Tune)
- روی دستگاه‌های لبه (Edge Devices) قابل‌استفاده باشند
- برای وظایف محدود کنترل‌پذیرتر باشند (More Controllable for Narrow Tasks)
- برای بارهای کاری پرتعداد عملی باشند (Practical for High-Volume Workloads)

مدل‌های تخصصی (Specialized Models) می‌توانند در وظایف محدود، اگر خوب آموزش داده یا تنظیم دقیق شوند، از مدل‌های عمومی بزرگ‌تر (Larger General Models) بهتر عمل کنند.

نمونه‌ها شامل موارد زیر هستند:

- مدل‌های امبدینگ (Embedding Models)
- بازرتبه‌بندها (Rerankers)
- دستیارهای کدنویسی (Code Assistants)
- تولیدکننده‌های SQL (SQL Generators)
- دسته‌بندهای سند (Document Classifiers)
- مدل‌های استخراج (Extraction Models)
- مدل‌های تعدیل محتوا (Moderation Models)
- چت‌بات‌های دامنه‌ای (Domain Chatbots)
- مؤلفه‌های گفتار یا OCR (Speech or OCR Components)

یک سامانهٔ عملی هوش مصنوعی (Practical AI System) ممکن است از مدل‌های زیادی استفاده کند:

```text
router
+ retriever
+ reranker
+ generator
+ verifier
+ safety classifier
+ embedding model
```
آینده احتمالاً شامل سبدهای مدل (Model Portfolios) خواهد بود، نه یک مدل جهانی (Universal Model) برای هر کار.

---

## ۱۴. فشرده‌سازی مدل (Model Compression)

تکنیک‌های فشرده‌سازی (Compression Techniques) هزینه را کاهش می‌دهند و قابلیت استقرار (Deployability) را بهبود می‌بخشند.

روش‌های مهم شامل موارد زیر هستند:

- کوانتیزاسیون (Quantization)
- هرس‌کردن (Pruning)
- تقطیر (Distillation)
- سازگاری کم‌رتبه (Low-Rank Adaptation)
- تنکی (Sparsity)
- اشتراک‌گذاری وزن (Weight Sharing)
- رمزگشایی حدسی با مدل‌های پیش‌نویس (Speculative Decoding with Draft Models)
- بازطراحی معماری (Architecture Redesign)

تقطیر (Distillation) به‌ویژه مهم است. یک مدل معلم بزرگ‌تر (Larger Teacher Model) می‌تواند سیگنال‌های آموزشی (Training Signals) برای یک مدل دانش‌آموز کوچک‌تر (Smaller Student Model) تولید کند.

یک جریان کاری ساده‌شدهٔ تقطیر (Simplified Distillation Workflow) چنین است:

```text
teacher model produces outputs
-> student model trains to imitate or improve on them
-> student is evaluated on target tasks
```
فشرده‌سازی (Compression) باید با دقت ارزیابی شود، زیرا افت کیفیت (Quality Loss) ممکن است در حوزه‌های خاص ظاهر شود:

- استدلال (Reasoning)
- عملکرد چندزبانه (Multilingual Performance)
- دانش نادر (Rare Knowledge)
- قالب‌بندی (Formatting)
- استفاده از ابزار (Tool Use)
- یادآوری زمینهٔ بلند (Long-Context Recall)
- رفتار ایمنی (Safety Behavior)

یک مدل فشرده‌شده (Compressed Model) به‌صورت خودکار مدل کم‌کیفیت‌تری نیست، اما یک مدل متفاوت است و باید به‌عنوان چنین چیزی آزمون شود.

---

## ۱۵. معماری‌های جدید (New Architectures)

ترنسفورمرها (Transformers) غالب هستند، اما پژوهش دربارهٔ جایگزین‌ها و ترکیب‌ها (Alternatives and Hybrids) ادامه دارد.

انگیزه‌ها شامل موارد زیر هستند:

- کاهش هزینهٔ توجه درجه‌دوم (Reducing Quadratic Attention Cost)
- بهبود مقیاس‌پذیری زمینهٔ بلند (Improving Long-Context Scaling)
- کاهش تأخیر استنتاج (Lowering Inference Latency)
- کاهش اندازهٔ کش KV (Reducing KV Cache Size)
- بهبود کارایی حافظه (Improving Memory Efficiency)
- امکان‌پذیر کردن بازگشت بهتر (Enabling Better Recurrence)
- بهبود بهره‌برداری از سخت‌افزار (Improving Hardware Utilization)

جهت‌های معماری بالقوه (Potential Architectural Directions) شامل موارد زیر هستند:

- مدل‌های فضای حالت (State-Space Models)
- مدل‌های دنباله‌ای بازگشتی (Recurrent Sequence Models)
- توجه خطی (Linear Attention)
- توجه تُنُک (Sparse Attention)
- آمیخته‌ای از متخصصان (Mixture-of-Experts)
- معماری‌های تقویت‌شده با بازیابی (Retrieval-Augmented Architectures)
- مدل‌های تقویت‌شده با حافظه (Memory-Augmented Models)
- سامانه‌های ترکیبی ترنسفورمر (Hybrid Transformer Systems)

معماری‌های جدید (New Architectures) باید نه‌تنها در کیفیت بنچمارک (Benchmark Quality)، بلکه در موارد زیر نیز رقابت کنند:

- پایداری آموزش (Training Stability)
- کارایی سخت‌افزاری (Hardware Efficiency)
- پشتیبانی اکوسیستم (Ecosystem Support)
- عملکرد استنتاج (Inference Performance)
- رفتار تنظیم دقیق (Fine-Tuning Behavior)
- سازگاری ابزارها (Tooling Compatibility)

ترنسفورمرها (Transformers) قوی باقی می‌مانند، زیرا نه‌تنها دقیق هستند؛ بلکه خوب فهمیده شده‌اند، خوب بهینه شده‌اند، و خوب پشتیبانی می‌شوند.

---

## ۱۶. آمیخته‌ای از متخصصان (Mixture-of-Experts)

مدل‌های آمیخته‌ای از متخصصان (Mixture-of-Experts Models) تعداد پارامترها (Parameter Count) را افزایش می‌دهند، درحالی‌که برای هر توکن فقط بخشی از مدل را فعال می‌کنند.

یک لایهٔ ساده‌شدهٔ MoE به‌صورت زیر کار می‌کند:

```text
token representation
-> router
-> selected experts
-> expert outputs
-> combine outputs
```
مدل‌های MoE می‌توانند ظرفیت بالا (High Capacity) را با محاسبهٔ فعال کمتر (Lower Active Compute) نسبت به مدل‌های چگال (Dense Models) فراهم کنند.

مزایا:

- ظرفیت کل پارامتری بزرگ‌تر (Larger Total Parameter Capacity)
- محاسبهٔ فعال کمتر به‌ازای هر توکن (Lower Active Compute per Token)
- تخصصی‌شدن میان متخصصان (Specialization across Experts)
- مقیاس‌پذیری مطلوب در برخی رژیم‌ها (Favorable Scaling in Some Regimes)

چالش‌ها:

- ناپایداری مسیریابی (Routing Instability)
- متوازن‌سازی بار (Load Balancing)
- کم‌استفاده‌ماندن متخصصان (Expert Underuse)
- ارتباط همه‌به‌همه (All-to-All Communication)
- سامانه‌های آموزشی پیچیده‌تر (More Complex Training Systems)
- سرویس‌دهی پیچیده‌تر (More Complex Serving)
- جای‌گذاری متخصصان (Expert Placement)
- اشکال‌زدایی دشوارتر (Harder Debugging)

MoE احتمالاً همچنان مهم باقی خواهد ماند، به‌ویژه برای مدل‌های بزرگ‌مقیاس (Large-Scale Models)، اما پیچیدگی سامانه‌ها (Systems Complexity) را به‌طور قابل‌توجهی افزایش می‌دهد.

---

## ۱۷. شخصی‌سازی (Personalization)

سامانه‌های آیندهٔ مدل‌های زبانی بزرگ (Future LLM Systems) ممکن است به‌طور مؤثرتری با کاربران، سازمان‌ها، دامنه‌ها، و جریان‌های کاری سازگار شوند.

شخصی‌سازی (Personalization) ممکن است شامل موارد زیر باشد:

- ترجیحات کاربر (User Preferences)
- سبک نوشتار (Writing Style)
- اصطلاحات دامنه (Domain Terminology)
- مجوزهای ابزار (Tool Permissions)
- زمینهٔ تاریخی (Historical Context)
- دانش سازمانی (Organizational Knowledge)
- جریان‌های کاری خاص وظیفه (Task-Specific Workflows)
- سامانه‌های حافظه (Memory Systems)

شخصی‌سازی (Personalization) می‌تواند سودمندی (Usefulness) را بهبود دهد، اما محدودیت‌های مهمی ایجاد می‌کند:

- حریم خصوصی (Privacy)
- رضایت (Consent)
- نگهداشت داده (Data Retention)
- کنترل کاربر (User Control)
- شفافیت (Transparency)
- ارزیابی (Evaluation)
- ایمنی (Safety)
- جداسازی میان کاربران یا مستأجرها (Separation between Users or Tenants)

یک سامانهٔ شخصی‌سازی خوب (Good Personalization System) باید صریح و کنترل‌پذیر باشد. کاربران و سازمان‌ها باید بفهمند چه چیزی ذخیره می‌شود، چگونه استفاده می‌شود، و چگونه می‌توان آن را حذف کرد.

---

## ۱۸. ارزیابی محوری‌تر خواهد شد (Evaluation Will Become More Central)

با توانمندتر شدن مدل‌ها، ارزیابی (Evaluation) دشوارتر می‌شود.

امتیازهای سادهٔ بنچمارک (Simple Benchmark Scores) کافی نیستند. مدل‌ها ممکن است در آزمون‌های ایستا (Static Tests) خوب عمل کنند، اما در جریان‌های کاری واقعی (Real Workflows) شکست بخورند.

ارزیابی آینده (Future Evaluation) باید موارد زیر را پوشش دهد:

- موفقیت وظیفه (Task Success)
- واقعیت‌مندی (Factuality)
- قابلیت اعتماد استدلال (Reasoning Reliability)
- کالیبراسیون (Calibration)
- استحکام (Robustness)
- عملکرد چندزبانه (Multilingual Performance)
- استفاده از زمینهٔ بلند (Long-Context Use)
- درستی فراخوانی ابزار (Tool-Call Correctness)
- رفتار عامل (Agent Behavior)
- ایمنی (Safety)
- حریم خصوصی (Privacy)
- تأخیر (Latency)
- هزینه (Cost)
- ریسک پس‌رفت (Regression Risk)

ارزیابی به‌طور فزاینده شبیه آزمون نرم‌افزار به‌علاوهٔ اندازه‌گیری علمی (Software Testing plus Scientific Measurement) خواهد شد.

یک سامانهٔ ارزیابی قوی (Strong Evaluation System) باید شامل موارد زیر باشد:

```text
unit tests
+ regression tests
+ benchmark suites
+ human review
+ adversarial tests
+ production monitoring
+ cost and latency metrics
```
سامانهٔ مستقرشده (Deployed System) باید ارزیابی شود، نه فقط مدل پایه (Base Model).

---

## ۱۹. از بنچمارک‌ها تا وظایف واقعی (From Benchmarks to Real Tasks)

بنچمارک‌ها (Benchmarks) مفید هستند، اما وظایف واقعی آشفته‌ترند.

ورودی‌های تولید (Production Inputs) شامل موارد زیر هستند:

- درخواست‌های مبهم (Ambiguous Requests)
- زمینهٔ ناقص (Incomplete Context)
- اسناد نویزی (Noisy Documents)
- محتوای خصمانه (Adversarial Content)
- زبان‌های ترکیبی (Mixed Languages)
- اصطلاحات دامنه‌ای (Domain-Specific Terminology)
- فایل‌های بدشکل (Malformed Files)
- تاریخچه‌های طولانی (Long Histories)
- شکست ابزارهای خارجی (External Tool Failures)

مدلی که روی یک بنچمارک خوب عمل می‌کند، ممکن است همچنان هنگام یکپارچه‌شدن در یک جریان کاری (Workflow) شکست بخورد.

ارزیابی وظیفهٔ واقعی (Real-Task Evaluation) باید اندازه‌گیری کند:

- آیا هدف کاربر کامل شد؟
- آیا خروجی درست بود؟
- آیا مدل به‌طور مناسب نامطمئن بود؟
- آیا سامانه ابزارها را درست استفاده کرد؟
- آیا در صورت نیاز منابع را ارجاع داد؟
- آیا از اقدامات ناایمن اجتناب کرد؟
- آیا تأخیر قابل‌قبول بود؟
- آیا هزینه قابل‌قبول بود؟

این موضوع ارزیابی را از امتیازدهی پاسخ‌های جداافتاده (Isolated Answer Scoring) به قابلیت اعتماد سرتاسری سامانه (End-to-End System Reliability) منتقل می‌کند.

---

## ۲۰. ایمنی، هم‌راستاسازی، و کنترل (Safety, Alignment, and Control)

با توانمندتر شدن مدل‌ها، کنترل رفتار (Controlling Behavior) مهم‌تر می‌شود.

ایمنی و هم‌راستاسازی (Safety and Alignment) شامل موارد زیر است:

- پیروی از دستورالعمل (Instruction Following)
- رفتار امتناع (Refusal Behavior)
- استحکام در برابر سوءاستفاده (Robustness to Misuse)
- حفاظت از حریم خصوصی (Privacy Protection)
- محدودیت‌های استفاده از ابزار (Tool-Use Constraints)
- اجتناب از خروجی‌های زیان‌بار (Avoidance of Harmful Outputs)
- مقاومت در برابر تزریق پرامپت (Resistance to Prompt Injection)
- عدم‌قطعیت کالیبره‌شده (Calibrated Uncertainty)
- سبک و لحن کنترل‌پذیر (Controllable Style and Tone)
- انطباق با سیاست کاربرد (Compliance with Application Policy)

این اهداف فقط در مرحلهٔ پس‌آموزش (Post-Training) حل نمی‌شوند. آن‌ها به کنترل‌های لایه‌ای (Layered Controls) نیاز دارند:

```text
training data
+ post-training
+ system prompts
+ retrieval controls
+ tool permissions
+ runtime filters
+ monitoring
+ human oversight where needed
```
هم‌راستاسازی خوب (Good Alignment) فقط دربارهٔ چیزی نیست که مدل می‌گوید. بلکه دربارهٔ این نیز هست که سامانهٔ کامل (Complete System) مجاز است چه کاری انجام دهد.

---

## ۲۱. حکمرانی بدون جایگزین‌کردن مهندسی (Governance Without Replacing Engineering)

نیازمندی‌های حکمرانی (Governance)، سیاست (Policy)، و انطباق (Compliance) ممکن است شکل ساخت سامانه‌های مدل زبانی بزرگ (LLM Systems) را تعیین کنند. بااین‌حال، قابلیت اعتماد عملی (Practical Reliability) همچنان به کنترل‌های مهندسی مشخص (Concrete Engineering Controls) وابسته است.

نمونه‌ها شامل موارد زیر هستند:

- منشأ مجموعه‌داده (Dataset Provenance)
- کنترل دسترسی (Access Control)
- لاگ‌های ممیزی (Audit Logs)
- کارت‌های مدل (Model Cards)
- گزارش‌های ارزیابی (Evaluation Reports)
- پاسخ به رخداد (Incident Response)
- سیاست‌های نگهداشت (Retention Policies)
- بازبینی استقرار (Deployment Review)
- آزمون رد-تیم (Red-Team Testing)
- پایش و بازگشت (Monitoring and Rollback)

پرسش مهندسی (Engineering Question) این است:

```text
What evidence shows that this model system behaves acceptably for its intended use?

That evidence should be produced through testing, documentation, monitoring, and controlled deployment practices.
```
---

## ۲۲. حریم خصوصی و محرمانگی (Privacy and Confidentiality)

سامانه‌های مدل زبانی بزرگ (LLM Systems) اغلب با داده‌های حساس (Sensitive Data) تعامل دارند.

سامانه‌های آینده به شیوه‌های قوی‌تر حریم خصوصی (Stronger Privacy Practices) نیاز خواهند داشت، از جمله:

- کمینه‌سازی داده (Data Minimization)
- استقرار محلی یا خصوصی (Local or Private Deployment)
- رمزنگاری در انتقال و در حالت سکون (Encryption in Transit and at Rest)
- ثبت دسترسی (Access Logging)
- جداسازی مستأجرها (Tenant Isolation)
- مدیریت امن پرامپت (Secure Prompt Handling)
- اجرای امن ابزار (Secure Tool Execution)
- حذف/پوشاندن اطلاعات در لاگ‌ها (Redaction in Logs)
- کنترل‌های نگهداشت (Retention Controls)
- جریان‌های کاری حذف (Deletion Workflows)
- تحلیل‌های حافظ حریم خصوصی (Privacy-Preserving Analytics)

حریم خصوصی (Privacy) باید در سراسر سامانهٔ کامل (Full System) در نظر گرفته شود:

```text
prompt
-> retrieval
-> model inference
-> tool calls
-> logs
-> monitoring
-> stored outputs
-> feedback data
```
یک استقرار خصوصی مدل (Private Model Deployment) همچنان می‌تواند از طریق لاگ‌ها (Logs)، کش‌ها (Caches)، فراخوانی‌های ابزار (Tool Calls)، یا کنترل دسترسی ضعیف (Poor Access Controls) داده نشت دهد.

---

## ۲۳. مدل‌های باز و استقرار محلی (Open Models and Local Deployment)

مدل‌های با وزن باز (Open-Weight Models) احتمالاً بخش مهمی از اکوسیستم (Ecosystem) باقی خواهند ماند.

آن‌ها از موارد زیر پشتیبانی می‌کنند:

- شفافیت پژوهشی (Research Transparency)
- آزمایش محلی (Local Experimentation)
- استقرار خصوصی (Private Deployment)
- تنظیم دقیق دامنه‌ای (Domain Fine-Tuning)
- کنترل هزینه (Cost Control)
- سفارشی‌سازی (Customization)
- استفادهٔ آفلاین (Offline Use)
- آموزش (Education)
- نوآوری اکوسیستم (Ecosystem Innovation)

استقرار محلی (Local Deployment) به‌ویژه زمانی مفید است که:

- داده نمی‌تواند از یک محیط خارج شود
- تأخیر باید کنترل شود
- هزینه باید قابل‌پیش‌بینی باشد
- سفارشی‌سازی مهم است
- وابستگی به APIهای خارجی نامطلوب است

بااین‌حال، استقرار محلی (Local Deployment) همچنین نیازمند مالکیت موارد زیر است:

- انتخاب مدل (Model Selection)
- زیرساخت استنتاج (Inference Infrastructure)
- امنیت (Security)
- به‌روزرسانی‌ها (Updates)
- ارزیابی (Evaluation)
- پایش (Monitoring)
- مدیریت هزینه (Cost Management)

مدل‌های باز (Open Models) وابستگی به ارائه‌دهندگان راه‌دور (Remote Providers) را کاهش می‌دهند، اما مسئولیت مهندسی (Engineering Responsibility) را حذف نمی‌کنند.

---

## ۲۴. مدل‌های زبانی بزرگ سازمانی و دامنه‌محور (Enterprise and Domain-Specific LLMs)

بسیاری از کاربردهای باارزش مدل‌های زبانی بزرگ (High-Value LLM Applications) دامنه‌محور (Domain-Specific) هستند.

نمونه‌ها شامل موارد زیر هستند:

- تحلیل اسناد حقوقی (Legal Document Analysis)
- مرور ادبیات پزشکی (Medical Literature Review)
- پژوهش مالی (Financial Research)
- مهندسی نرم‌افزار (Software Engineering)
- پشتیبانی مشتری (Customer Support)
- کشف علمی (Scientific Discovery)
- آموزش (Education)
- عملیات تولید (Manufacturing Operations)
- دستیارهای دانش داخلی (Internal Knowledge Assistants)

سامانه‌های دامنه‌ای (Domain Systems) اغلب به موارد زیر نیاز دارند:

- بازیابی قوی (Strong Retrieval)
- اصطلاحات کنترل‌شده (Controlled Terminology)
- واقعیت‌مندی بالا (High Factuality)
- پاسخ‌های قابل‌ممیزی (Auditable Answers)
- کنترل‌های حریم خصوصی (Privacy Controls)
- یکپارچه‌سازی با ابزارهای موجود (Integration with Existing Tools)
- ارزیابی خاص جریان کاری (Workflow-Specific Evaluation)
- بازبینی انسانی برای تصمیم‌های حساس (Human Review for Critical Decisions)

برای بسیاری از سازمان‌ها، رویکرد برنده این نیست که یک مدل مرزی (Frontier Model) را از ابتدا آموزش دهند. بلکه ترکیب یک مدل پایهٔ توانمند (Capable Base Model) با دادهٔ دامنه‌ای (Domain Data)، بازیابی (Retrieval)، آداپتورها (Adapters)، ابزارها (Tools)، ارزیابی (Evaluation)، و حکمرانی (Governance) است.

---

## ۲۵. سامانه‌های انسان در حلقه (Human-in-the-Loop Systems)

مدل‌های زبانی بزرگ (LLMs) زمانی قابل‌اعتمادتر هستند که برای وظایف پراثر (High-Impact Tasks) با نظارت انسانی مناسب (Appropriate Human Oversight) یکپارچه شوند.

الگوهای انسان در حلقه (Human-in-the-Loop Patterns) شامل موارد زیر هستند:

- بازبینی پیش از اقدام (Review before Action)
- تأیید برای اجرای ابزار (Approval for Tool Execution)
- اعتبارسنجی متخصص (Expert Validation)
- جمع‌آوری بازخورد (Feedback Collection)
- مسیرهای ارجاع/تصعید (Escalation Paths)
- جریان‌های کاری اصلاح (Correction Workflows)
- یادگیری فعال (Active Learning)
- نمونه‌گیری ممیزی (Audit Sampling)

هدف این نیست که انسان‌ها در هر حلقه‌ای قرار گیرند. هدف این است که قضاوت انسانی (Human Judgment) در جاهایی قرار گیرد که خطاهای مدل پرهزینه، مبهم، یا به‌سختی قابل‌تشخیص خودکار هستند.

یک سامانهٔ خوب تعریف می‌کند:

```text
what the model can do alone
what requires confirmation
what must be escalated
what is never allowed
```
---

## ۲۶. مدل‌های زبانی بزرگ به‌عنوان رابط‌ها (LLMs as Interfaces)

مدل‌های زبانی (Language Models) در حال تبدیل‌شدن به رابط‌هایی برای نرم‌افزار، داده، و جریان‌های کاری هستند.

به‌جای اینکه کاربران مجبور باشند هر منو، زبان پرس‌وجو (Query Language)، یا API را یاد بگیرند، یک مدل زبانی بزرگ (LLM) می‌تواند زبان طبیعی (Natural Language) را به اقدام‌ها ترجمه کند.

نمونه‌ها شامل موارد زیر هستند:

- پرسیدن سؤال روی پایگاه‌های داده (Asking Questions over Databases)
- کنترل نرم‌افزار کسب‌وکار (Controlling Business Software)
- تولید گزارش‌ها (Generating Reports)
- پیمایش اسناد (Navigating Documents)
- نوشتن و اجرای کد (Writing and Running Code)
- هماهنگ‌کردن جریان‌های کاری چندمرحله‌ای (Coordinating Multi-Step Workflows)
- خلاصه‌سازی وضعیت سامانه (Summarizing System State)

این موضوع طراحی محصول (Product Design) را تغییر می‌دهد. مدل فقط یک چت‌بات (Chatbot) نیست؛ بلکه به یک لایهٔ کنترل (Control Layer) تبدیل می‌شود.

رابط (Interface) همچنان باید قابل‌اعتماد باشد:

- نشان دهد چه اقدامی انجام خواهد شد
- در صورت نیاز تأیید بخواهد
- دادهٔ منبع را آشکار کند
- ابهام را مدیریت کند
- از خطاها بازیابی شود
- از اقدامات غیرمجاز جلوگیری کند
- لاگ‌ها را نگه دارد

زبان طبیعی (Natural Language) انعطاف‌پذیر است، اما سامانه‌های تولیدی (Production Systems) در زیرساخت خود به اجرای ساخت‌یافته (Structured Execution) نیاز دارند.

---

## ۲۷. مهندسی نرم‌افزار با مدل‌های زبانی بزرگ (Software Engineering with LLMs)

مدل‌های زبانی بزرگ (LLMs) جریان‌های کاری مهندسی نرم‌افزار (Software Engineering Workflows) را تغییر می‌دهند.

آن‌ها می‌توانند در موارد زیر کمک کنند:

- تولید کد (Code Generation)
- توضیح کد (Code Explanation)
- ساخت آزمون (Test Creation)
- بازآرایی کد (Refactoring)
- مستندسازی (Documentation)
- اشکال‌زدایی (Debugging)
- مهاجرت (Migration)
- بازبینی کد (Code Review)
- کاوش API (API Exploration)
- پیمایش مخزن (Repository Navigation)

قوی‌ترین سامانه‌ها مدل‌های زبانی بزرگ (LLMs) را با ابزارها (Tools) ترکیب می‌کنند:

```text
repository search
+ static analysis
+ tests
+ type checking
+ execution
+ version control
+ code review
```
عامل‌های کدنویسی (Code Agents) باید متفاوت از دستیارهای چت (Chat Assistants) ارزیابی شوند. خروجی فقط متن نیست؛ بلکه تغییری در یک کدبیس (Codebase) است.

معیارهای مهم شامل موارد زیر هستند:

- آزمون‌های پاس‌شده (Tests Passed)
- باگ‌های ایجادشده (Bugs Introduced)
- نگهداشت‌پذیری (Maintainability)
- امنیت (Security)
- سازگاری سبک (Style Consistency)
- حداقلی‌بودن تغییرات (Minimality of Changes)
- توانایی بازیابی از شکست‌ها (Ability to Recover from Failures)
- درستی استفاده از ابزار (Correctness of Tool Use)

مهندسی نرم‌افزار با کمک مدل‌های زبانی بزرگ (LLM-Assisted Software Engineering) احتمالاً به یکی از مهم‌ترین دامنه‌های کاربردی (Applied Domains) تبدیل خواهد شد.

---

## ۲۸. کشف علمی و فنی (Scientific and Technical Discovery)

مدل‌های زبانی بزرگ (LLMs) ممکن است با کمک در موارد زیر، کار فنی (Technical Work) را تسریع کنند:

- مرور ادبیات (Literature Review)
- تولید فرضیه (Hypothesis Generation)
- برنامه‌ریزی آزمایش (Experiment Planning)
- پیاده‌سازی کد (Code Implementation)
- تحلیل داده (Data Analysis)
- دست‌کاری نمادین (Symbolic Manipulation)
- کنترل شبیه‌سازی (Simulation Control)
- تفسیر نتایج (Result Interpretation)
- گزارش‌نویسی (Report Writing)

ارزش آن‌ها زمانی افزایش می‌یابد که به ابزارها (Tools) متصل شوند:

- سامانه‌های جست‌وجو (Search Systems)
- پایگاه‌های داده (Databases)
- نرم‌افزار آزمایشگاه (Laboratory Software)
- اثبات‌گرهای قضیه (Theorem Provers)
- حل‌گرهای عددی (Numerical Solvers)
- نوت‌بوک‌ها (Notebooks)
- ابزارهای بصری‌سازی (Visualization Tools)

اما استفادهٔ علمی (Scientific Use) نیازمند احتیاط است. توضیحات محتمل (Plausible Explanations) شواهد (Evidence) نیستند. فرضیه‌های تولیدشده (Generated Hypotheses) نیازمند راستی‌آزمایی (Verification) هستند.

یک دستیار علمی مفید (Useful Scientific Assistant) باید:

- منابع را ارجاع دهد (Cite Sources)
- شواهد را از حدس و گمان جدا کند (Separate Evidence from Speculation)
- در صورت امکان محاسبات را اجرا کند (Run Calculations when Possible)
- عدم‌قطعیت را آشکار کند (Expose Uncertainty)
- از بازتولیدپذیری پشتیبانی کند (Support Reproducibility)
- سوابق آزمایش را حفظ کند (Preserve Experimental Records)

مدل می‌تواند کار استدلالی (Reasoning Work) را تسریع کند، اما جایگزین اعتبارسنجی تجربی (Empirical Validation) نمی‌شود.

---

## ۲۹. محدودیت‌های اقتصادی (Economic Constraints)

سامانه‌های آیندهٔ مدل‌های زبانی بزرگ (Future LLM Systems) توسط اقتصاد (Economics) شکل خواهند گرفت.

محرک‌های هزینهٔ مهم (Important Cost Drivers) شامل موارد زیر هستند:

- محاسبهٔ پیش‌آموزش (Pretraining Compute)
- دادهٔ پس‌آموزش (Post-Training Data)
- ارزیابی (Evaluation)
- سخت‌افزار استنتاج (Inference Hardware)
- پهنای باند حافظه (Memory Bandwidth)
- طول زمینه (Context Length)
- طول خروجی (Output Length)
- فراخوانی‌های ابزار (Tool Calls)
- نیروی کار مهندسی (Engineering Labor)
- پایش و نگهداشت (Monitoring and Maintenance)
- آزمایش‌های شکست‌خورده (Failed Experiments)
- انطباق و بازبینی (Compliance and Review)

مدلی که اندکی توانمندتر اما بسیار گران‌تر است، ممکن است انتخاب مهندسی درست (Right Engineering Choice) نباشد.

هدف عملی (Practical Objective) این است:

```text
maximize useful capability per unit cost
```
This favors:

- smaller specialized models
- caching
- quantization
- routing
- retrieval
- distillation
- efficient decoding
- task-specific evaluation
- clear product constraints

فشار اقتصادی (Economic Pressure) همچنان پژوهش در کارایی (Efficiency Research) را پیش خواهد برد.

---

## ۳۰. محدودیت‌های زیست‌محیطی و سخت‌افزاری (Environmental and Hardware Constraints)

پیشرفت مدل‌های زبانی بزرگ (LLM Progress) به سخت‌افزار (Hardware)، انرژی (Energy)، خنک‌سازی (Cooling)، شبکه‌سازی (Networking)، و زنجیره‌های تأمین (Supply Chains) وابسته است.

محدودیت‌های مهم شامل موارد زیر هستند:

- دسترس‌پذیری شتاب‌دهنده‌ها (Accelerator Availability)
- ظرفیت حافظه (Memory Capacity)
- پهنای باند حافظه (Memory Bandwidth)
- عملکرد اتصال بین‌گره‌ای (Interconnect Performance)
- برق مرکز داده (Datacenter Power)
- ظرفیت خنک‌سازی (Cooling Capacity)
- قابلیت اطمینان سخت‌افزار (Hardware Reliability)
- توان عملیاتی ذخیره‌سازی (Storage Throughput)
- توپولوژی شبکه (Network Topology)

این محدودیت‌ها بر طراحی مدل (Model Design) اثر می‌گذارند.

برای مثال:

- مدل‌های زمینهٔ بلند (Long-Context Models) فشار حافظه (Memory Pressure) را افزایش می‌دهند
- مدل‌های MoE پیچیدگی ارتباطات (Communication Complexity) را افزایش می‌دهند
- مدل‌های بزرگ‌تر هزینهٔ سرویس‌دهی (Serving Cost) را افزایش می‌دهند
- کوانتیزاسیون (Quantization) نیازهای حافظه و پهنای باند (Memory and Bandwidth Needs) را کاهش می‌دهد
- معماری‌های تُنُک یا ترکیبی (Sparse or Hybrid Architectures) ممکن است کارایی را بهبود دهند

توسعهٔ آیندهٔ مدل‌ها همچنان توسط این‌که چه چیزی روی سخت‌افزار به‌صورت کارآمد اجرا می‌شود، شکل خواهد گرفت.

---

## ۳۱. مهارت‌های عملی برای مهندسان مدل‌های زبانی بزرگ (Practical Skills for LLM Engineers)

این حوزه به مهندسانی پاداش می‌دهد که هم مدل‌سازی (Modeling) و هم سامانه‌ها (Systems) را درک می‌کنند.

مهارت‌های مهم شامل موارد زیر هستند:

- طراحی خط لولهٔ داده (Data Pipeline Design)
- رفتار توکنایزر (Tokenizer Behavior)
- معماری ترنسفورمر (Transformer Architecture)
- بهینه‌سازی و پایداری آموزش (Optimization and Training Stability)
- آموزش توزیع‌شده (Distributed Training)
- طراحی ارزیابی (Evaluation Design)
- سامانه‌های بازیابی (Retrieval Systems)
- بهینه‌سازی استنتاج (Inference Optimization)
- کوانتیزاسیون (Quantization)
- مهندسی پرامپت و زمینه (Prompt and Context Engineering)
- یکپارچه‌سازی ابزار (Tool Integration)
- پایش و مشاهده‌پذیری (Monitoring and Observability)
- امنیت و حریم خصوصی (Security and Privacy)
- اندازه‌گیری محصول‌محور (Product-Oriented Measurement)

مهندسی مدل‌های زبانی بزرگ (LLM Engineering) میان‌رشته‌ای (Interdisciplinary) است. مدل فقط یک بخش از یک سامانهٔ کاری (Working System) است.

یک مهندس قوی می‌تواند میان لایه‌ها حرکت کند:

```text
data
-> model
-> training
-> evaluation
-> serving
-> application
-> monitoring
```
---

## ۳۲. بعد از این چه بیاموزیم (What to Learn Next)

پس از این کتابچه (Handbook)، موضوعات بعدی مفید شامل موارد زیر هستند:

### مدل‌سازی (Modeling)

- گونه‌های ترنسفورمر (Transformer Variants)
- توجه کارآمد (Efficient Attention)
- مدل‌های فضای حالت (State-Space Models)
- آمیخته‌ای از متخصصان (Mixture-of-Experts)
- معماری‌های چندوجهی (Multimodal Architectures)
- آموزش استدلال‌محور (Reasoning-Oriented Training)

### آموزش (Training)

- آموزش توزیع‌شدهٔ بزرگ‌مقیاس (Large-Scale Distributed Training)
- گزینش و پالایش مجموعه‌داده (Dataset Curation)
- تولید دادهٔ مصنوعی (Synthetic Data Generation)
- بهینه‌سازی ترجیح (Preference Optimization)
- یادگیری تقویتی برای مدل‌های زبانی (Reinforcement Learning for Language Models)
- یادگیری پیوسته (Continual Learning)

### بازیابی و عامل‌ها (Retrieval and Agents)

- جست‌وجوی ترکیبی (Hybrid Search)
- بازرتبه‌بندی (Reranking)
- GraphRAG
- فراخوانی ابزار (Tool Calling)
- ارزیابی عامل (Agent Evaluation)
- سامانه‌های حافظه (Memory Systems)
- دفاع در برابر تزریق پرامپت (Prompt Injection Defense)

### سامانه‌ها (Systems)

- سرورهای استنتاج (Inference Servers)
- مدیریت کش KV (KV Cache Management)
- کوانتیزاسیون (Quantization)
- رمزگشایی حدسی (Speculative Decoding)
- موازی‌سازی مدل (Model Parallelism)
- مشاهده‌پذیری (Observability)
- بهینه‌سازی هزینه (Cost Optimization)

### ارزیابی (Evaluation)

- طراحی بنچمارک (Benchmark Design)
- ارزیابی انسانی (Human Evaluation)
- مدل زبانی بزرگ به‌عنوان داور (LLM-as-a-Judge)
- بررسی واقعیت‌مندی (Factuality Checking)
- آزمون ایمنی (Safety Testing)
- مجموعه‌های پس‌رفت (Regression Suites)
- پایش تولید (Production Monitoring)

بهترین مسیر یادگیری، پروژه‌محور (Project-Driven) است. یک مورد کاربرد مشخص (Concrete Use Case) انتخاب کنید، خط لولهٔ کامل (Complete Pipeline) را بسازید، آن را ارزیابی کنید، و ضعیف‌ترین لایه را بهبود دهید.

---

## ۳۳. یک نقشه‌راه عملی (A Practical Roadmap)

یک نقشه‌راه عمل‌گرایانه (Pragmatic Roadmap) برای ساخت سامانه‌های مدل زبانی بزرگ (LLM Systems) چنین است:

```text
1. Start with a clear task.
2. Select a strong baseline model.
3. Build an evaluation set.
4. Add retrieval if knowledge freshness or grounding is needed.
5. Tune prompts and decoding.
6. Add tools only when they solve a real limitation.
7. Measure quality, latency, and cost.
8. Add safety and permission controls.
9. Monitor production behavior.
10. Iterate based on observed failures.
```
For most teams, this roadmap is more useful than immediately training a model from scratch.

Training from scratch is appropriate when there is a strong reason, such as:

- unique data
- special architecture requirements
- strict deployment constraints
- research goals
- large-scale product needs
- economic advantage at scale

Otherwise, adapting and integrating existing models is usually the more practical path.

---

## ۳۴. پرسش‌های پژوهشی باز (Open Research Questions)

بسیاری از پرسش‌های مهم همچنان باز هستند.

نمونه‌ها شامل موارد زیر هستند:

- مدل‌ها چگونه می‌توانند قابل‌اعتمادتر استدلال کنند؟
- توهم‌ها (Hallucinations) در وظایف واقعی چگونه می‌توانند کاهش یابند؟
- مدل‌های زمینهٔ بلند (Long-Context Models) چگونه می‌توانند از اطلاعات دور بهتر استفاده کنند؟
- عامل‌ها (Agents) چگونه می‌توانند به‌صورت سخت‌گیرانه ارزیابی شوند؟
- دادهٔ مصنوعی (Synthetic Data) چگونه می‌تواند بدون فروپاشی کیفیت (Quality Collapse) تولید شود؟
- مدل‌ها چگونه می‌توانند به‌صورت پیوسته بدون فراموشی (Forgetting) یاد بگیرند؟
- مدل‌های چندوجهی (Multimodal Models) چگونه می‌توانند خروجی‌ها را دقیق‌تر زمینه‌سازی کنند؟
- استنتاج (Inference) چگونه می‌تواند بسیار ارزان‌تر شود؟
- حریم خصوصی (Privacy) چگونه می‌تواند در طول آموزش و سرویس‌دهی حفظ شود؟
- چگونه می‌توان رفتار مدل (Model Behavior) را تفسیرپذیرتر کرد؟
- چگونه می‌توان راستی‌آزماهای قابل‌اعتماد (Trustworthy Verifiers) ساخت؟
- مدل‌ها چگونه می‌توانند عدم‌قطعیت (Uncertainty) را بهتر مدیریت کنند؟

این‌ها فقط پرسش‌های پژوهشی (Research Questions) نیستند. آن‌ها همچنین پرسش‌های مهندسی (Engineering Questions) هستند که بر استقرارهای واقعی (Real Deployments) اثر می‌گذارند.

---

## ۳۵. برداشت‌های نادرست رایج دربارهٔ آینده (Common Misconceptions About the Future)

### برداشت نادرست 1: مدل‌های بزرگ‌تر همه‌چیز را حل خواهند کرد (Misconception 1: Larger models will solve everything)

مدل‌های بزرگ‌تر کمک می‌کنند، اما به‌صورت خودکار زمینه‌سازی (Grounding)، حریم خصوصی (Privacy)، تأخیر (Latency)، هزینه (Cost)، ایمنی ابزار (Tool Safety)، یا قابلیت اعتماد کاربرد (Application Reliability) را حل نمی‌کنند.

### برداشت نادرست 2: زمینهٔ بلند نیاز به بازیابی را حذف می‌کند (Misconception 2: Long context removes the need for retrieval)

زمینهٔ بلند (Long Context) کمک می‌کند، اما بازیابی (Retrieval) همچنان برای مرتبط‌بودن (Relevance)، تازگی (Freshness)، کنترل هزینه (Cost Control)، و نسبت‌دادن منبع (Source Attribution) لازم است.

### برداشت نادرست 3: عامل‌ها فقط به پرامپت‌های بهتر نیاز دارند (Misconception 3: Agents only need better prompts)

عامل‌های قابل‌اعتماد (Reliable Agents) به ابزارها (Tools)، مدیریت وضعیت (State Management)، مجوزها (Permissions)، اعتبارسنجی (Validation)، ارزیابی (Evaluation)، و پایش (Monitoring) نیاز دارند.

### برداشت نادرست 4: بنچمارک‌ها قابلیت را کامل اندازه‌گیری می‌کنند (Misconception 4: Benchmarks fully measure capability)

بنچمارک‌ها (Benchmarks) سیگنال‌های مفیدی هستند، اما جریان‌های کاری واقعی (Real Workflows) به ارزیابی سرتاسری (End-to-End Evaluation) نیاز دارند.

### برداشت نادرست 5: مدل‌های باز پیچیدگی زیرساخت را حذف می‌کنند (Misconception 5: Open models eliminate infrastructure complexity)

مدل‌های باز (Open Models) کنترل فراهم می‌کنند، اما سرویس‌دهی (Serving)، امنیت (Security)، ارزیابی (Evaluation)، به‌روزرسانی‌ها (Updates)، و پایش (Monitoring) همچنان ضروری هستند.

### برداشت نادرست 6: کوانتیزاسیون فقط یک جزئیات استقرار است (Misconception 6: Quantization is only a deployment detail)

کوانتیزاسیون (Quantization) می‌تواند بر رفتار مدل (Model Behavior) اثر بگذارد. باید مانند هر تغییر مدل دیگری ارزیابی شود.

---

## ۳۶. طراحی برای تغییر (Designing for Change)

سامانه‌های مدل زبانی بزرگ (LLM Systems) باید با در نظر گرفتن تغییر طراحی شوند.

مدل‌ها تغییر خواهند کرد. توکنایزرها (Tokenizers) تغییر خواهند کرد. طول‌های زمینه (Context Lengths) تغییر خواهند کرد. زمان‌اجرای استنتاج (Inference Runtimes) تغییر خواهد کرد. روش‌های ارزیابی (Evaluation Methods) تغییر خواهند کرد. انتظارات کاربران (User Expectations) تغییر خواهد کرد.

یک سامانهٔ قابل‌نگهداشت مدل زبانی بزرگ (Maintainable LLM System) باید مؤلفه‌های زیر را صریح و قابل‌جایگزینی کند:

- ارائه‌دهندهٔ مدل یا وزن‌های مدل (Model Provider or Model Weights)
- توکنایزر (Tokenizer)
- قالب چت (Chat Template)
- ساخت پرامپت (Prompt Construction)
- خط لولهٔ بازیابی (Retrieval Pipeline)
- بازرتبه‌بند (Reranker)
- شِماهای ابزار (Tool Schemas)
- پارامترهای رمزگشایی (Decoding Parameters)
- فیلترهای ایمنی (Safety Filters)
- مجموعه‌های ارزیابی (Evaluation Sets)
- داشبوردهای پایش (Monitoring Dashboards)
- پیکربندی استقرار (Deployment Configuration)

سخت‌کدنویسی این قطعات (Hard-Coding these Pieces) تکرار و بهبود (Iteration) را دشوار و پرریسک می‌کند.

یک سامانهٔ خوب با پرامپت‌ها (Prompts)، مدل‌ها (Models)، تنظیمات بازیابی (Retrieval Settings)، و پارامترهای رمزگشایی (Decoding Parameters) به‌عنوان مصنوعات نسخه‌گذاری‌شده (Versioned Artifacts) رفتار می‌کند.

---

## ۳۷. ذهنیت مهندسی (The Engineering Mindset)

کار با مدل‌های زبانی بزرگ (LLM Work) از بیرون می‌تواند رازآلود به نظر برسد، اما سامانه‌های قابل‌اعتماد از طریق مهندسی منضبط (Disciplined Engineering) ساخته می‌شوند.

ذهنیت (Mindset) این است:

```text
define the task
-> establish a baseline
-> measure behavior
-> identify failure modes
-> change one thing
-> evaluate again
-> deploy carefully
-> monitor continuously
```
این موضوع چه سامانه یک دستیار محلی کوچک (Small Local Assistant) باشد و چه یک پلتفرم تولیدی بزرگ (Large Production Platform)، صدق می‌کند.

رایج‌ترین شکست، نبود تکنیک‌های پیشرفته (Advanced Techniques) نیست. بلکه نبود اندازه‌گیری روشن (Clear Measurement) است.

بدون ارزیابی (Evaluation)، هر تغییری حدس‌وگمان (Guesswork) است.

---

## ۳۸. چک‌لیست نهایی (Final Checklist)

هنگام برنامه‌ریزی کارهای آیندهٔ مدل‌های زبانی بزرگ (Future LLM Work)، بررسی کنید:

- وظیفهٔ هدف (Target Task) به‌روشنی تعریف شده است.
- معیارهای موفقیت (Success Criteria) قابل‌اندازه‌گیری هستند.
- یک خط پایه (Baseline) وجود دارد.
- دادهٔ ارزیابی (Evaluation Data) کاربرد واقعی را بازتاب می‌دهد.
- انتخاب مدل (Model Choice) با کیفیت، تأخیر، و هزینه توجیه شده است.
- زمانی که زمینه‌سازی (Grounding) یا تازگی (Freshness) لازم است، از بازیابی (Retrieval) استفاده می‌شود.
- استفاده از ابزار (Tool Use) محدود و مشاهده‌پذیر است.
- پرامپت‌ها و قالب‌ها (Prompts and Templates) نسخه‌گذاری شده‌اند.
- تنظیمات رمزگشایی (Decoding Settings) ارزیابی شده‌اند.
- کوانتیزاسیون یا فشرده‌سازی (Quantization or Compression) روی وظایف هدف آزمون شده است.
- کنترل‌های ایمنی و حریم خصوصی (Safety and Privacy Controls) بخشی از طراحی هستند.
- پایش (Monitoring) کیفیت، تأخیر، هزینه، و شکست‌ها را پوشش می‌دهد.
- سامانه می‌تواند بدون بازنویسی‌های شکننده (Fragile Rewrites) به‌روزرسانی شود.
- بازبینی انسانی (Human Review) در جاهایی که خطاها اثر بالا دارند گنجانده شده است.
- هزینه به‌ازای نتیجهٔ مفید (Cost per Useful Outcome) اندازه‌گیری می‌شود.
- شکست‌های تولید (Production Failures) به ارزیابی بازخورانده می‌شوند.

---

## ۳۹. نکات کلیدی (Key Takeaways)

آیندهٔ مدل‌های زبانی بزرگ (LLMs) توسط قابلیت (Capability)، کارایی (Efficiency)، قابلیت اعتماد (Reliability)، و یکپارچه‌سازی (Integration) شکل خواهد گرفت.

مقیاس‌پذیری (Scaling) ادامه خواهد داشت، اما دادهٔ بهتر (Better Data)، استنتاج بهتر (Better Inference)، بازیابی قوی‌تر (Stronger Retrieval)، استفادهٔ قابل‌اعتماد از ابزار (Reliable Tool Use)، چندوجهی‌بودن (Multimodality)، فشرده‌سازی (Compression)، و ارزیابی (Evaluation) به همان اندازه اهمیت خواهند داشت.

مفیدترین سامانه‌ها اغلب چندین مؤلفه را ترکیب خواهند کرد: مدل‌های زبانی (Language Models)، بازیاب‌ها (Retrievers)، بازرتبه‌بندها (Rerankers)، ابزارها (Tools)، راستی‌آزماها (Verifiers)، حافظه (Memory)، لایه‌های ایمنی (Safety Layers)، و پایش (Monitoring).

مهندسی مدل‌های زبانی بزرگ (LLM Engineering) از دموهای مدل (Model Demos) به سامانه‌های پایدار (Durable Systems) در حال حرکت است. مهارت مرکزی فقط دانستن نحوهٔ کار مدل‌ها نیست، بلکه دانستن این است که چگونه آن‌ها را تحت محدودیت‌های واقعی (Real Constraints) مفید کنیم.

این حوزه همچنان تغییر خواهد کرد. اصول این کتابچه (Handbook) نقطه‌های شروع عملی باقی می‌مانند: داده را بفهمید، مدل را بفهمید، رفتار را اندازه‌گیری کنید، سامانه را کنترل کنید، و بر اساس شواهد (Evidence) بهبود دهید.

---

## پرسش‌های مرور (Review Questions)

1. کدام اصول مهندسی (Engineering Principles) در این کتاب احتمالاً حتی با تغییر معماری‌ها نیز مفید باقی می‌مانند؟
2. چرا عملکرد بهتر در بنچمارک (Benchmark Performance) تضمین‌کنندهٔ سامانه‌های واقعی بهتر (Better Real-World Systems) نیست؟
3. یک مهندس مدل زبانی بزرگ (LLM Engineer) فراتر از معماری مدل (Model Architecture) چه مهارت‌هایی باید توسعه دهد؟
---
[قبلی: سامانه‌ها (Systems)](./10-inference-and-decoding.md) |
[فهرست مطالب (Contents)](./index.md) |

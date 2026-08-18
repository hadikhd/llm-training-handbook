---
id: evaluation
title: ارزیابی
sidebar_label: ارزیابی
sidebar_position: 1
---
# فصل ۸: ارزیابی (Evaluation)
<div className="chapter-hero">

![فصل ۸: ارزیابی ](/static/img/chapters/evaluation.png)

</div>

[قبلی: پس‌آموزش](./07-post-training.md) |
[فهرست مطالب](./index.md) |
[بعدی: سیستم‌ها](./09-systems.md)

---

# پس‌آموزش برای تنظیم دقیق مدل‌های زبانی بزرگ (Post-Training for LLM Fine-Tuning)

پس‌آموزش (Post-training) مرحله‌ای است که در آن یک مدل زبانی ازپیش‌آموزش‌دیده (Pretrained Language Model) به یک دستیار قابل‌استفاده، مدل دامنه‌ای (Domain Model) یا سامانهٔ تخصصی‌شده برای وظیفه (Task-Specialized System) تبدیل می‌شود. پیش‌آموزش (Pretraining) ساختار آماری گسترده را از متن‌های بزرگ‌مقیاس به یک مدل می‌آموزد. پس‌آموزش (Post-training) تعیین می‌کند که این دانش چگونه بازیابی شود، چگونه قالب‌بندی شود، چگونه محدود شود، و چگونه با نیت کاربر هم‌راستا (Aligned) گردد.

یک مدل ذهنی مفید:

```text
pretraining -> base model
post-training -> instruction-following / preference-aligned / task-adapted model
```
پیش‌آموزش (Pretraining)، پیش‌بینی توکن بعدی (Next-Token Prediction) را روی پیکره‌های متنی گسترده (Broad Corpora) بهینه می‌کند. پس‌آموزش (Post-training)، آموزش را روی داده‌های محدودتر اما با سیگنال قوی‌تر ادامه می‌دهد، مانند دستورالعمل‌ها (Instructions)، گفتگوها (Conversations)، نمایش‌ها (Demonstrations)، ترجیحات انسانی (Human Preferences)، ردپاهای استفاده از ابزار (Tool-Use Traces)، یا مثال‌های خاص دامنه (Domain-Specific Examples).

پس‌آموزش (Post-training) معمولاً رفتار (Behavior) را بیش از دانش خام (Raw Knowledge) تغییر می‌دهد. این مرحله می‌تواند توانایی مدل را برای آشکارسازی دانشی که از قبل در پارامترهای آن بازنمایی شده است، پیروی از دستورالعمل‌ها، رد درخواست‌های ناایمن، حفظ قالب گفتگو (Conversational Format)، و ترجیح پاسخ‌های مفیدتر بهبود دهد. نباید با آن به‌عنوان جایگزینی قابل‌اعتماد برای پیش‌آموزش ضعیف یا پوشش ناکافی دامنه برخورد کرد.

روش‌های اصلی پس‌آموزش (Post-training) عبارت‌اند از:

- تنظیم دقیق نظارت‌شده (Supervised Fine-Tuning - SFT)
- تنظیم دقیق کارآمد از نظر پارامتر (Parameter-Efficient Fine-Tuning - PEFT)
- مدل‌سازی پاداش (Reward Modeling)
- یادگیری تقویتی از بازخورد انسانی (Reinforcement Learning from Human Feedback - RLHF)
- بهینه‌سازی سیاست مجاورتی (Proximal Policy Optimization - PPO)
- بهینه‌سازی مستقیم ترجیح (Direct Preference Optimization - DPO)

تنظیم با دستورالعمل (Instruction Tuning) اغلب به‌عنوان تنظیم دقیق نظارت‌شده (Supervised Fine-Tuning) توصیف می‌شود، زیرا مدل به‌طور صریح روی مجموعه‌داده‌های دستورالعملی (Instruction Datasets) تنظیم دقیق می‌شود. محتوای دوره این موضوع را مستقیماً بیان می‌کند: «تنظیم با دستورالعمل (Instruction Tuning) اغلب تنظیم دقیق نظارت‌شده (Supervised Fine-Tuning - SFT) نیز نامیده می‌شود» زیرا مدل با یک مجموعه‌داده از دستورالعمل‌ها تنظیم دقیق می‌شود.

---

## 1. مدل‌های پایه و مدل‌های پس‌آموزش‌دیده (Base Models and Post-Trained Models)

یک مدل پایه (Base Model) برای ادامه‌دادن متن آموزش داده می‌شود. با داشتن یک پیشوند (Prefix)، توکن‌های بعدی محتمل را پیش‌بینی می‌کند. این موضوع آن را انعطاف‌پذیر می‌کند، اما لزوماً مطیع (Obedient)، ایمن (Safe)، مختصر (Concise)، یا گفتگومحور (Conversational) نمی‌سازد.

برای مثال، اگر به یک مدل پایه بدهیم:

```text
Explain gradient descent.

may continue with a textbook paragraph, a list of unrelated examples, a training-data-like fragment, or even another prompt. It has not necessarily learned that the user expects a direct answer.
```
ممکن است با یک پاراگراف کتاب‌درسی، فهرستی از مثال‌های نامرتبط، قطعه‌ای شبیه دادهٔ آموزشی، یا حتی یک پرامپت دیگر ادامه دهد. این مدل لزوماً یاد نگرفته است که کاربر انتظار یک پاسخ مستقیم دارد.

یک مدل پس‌آموزش‌دیده (Post-Trained Model) بهینه شده است تا همان ورودی را به‌عنوان یک دستورالعمل (Instruction) تفسیر کند:

```text
User: Explain gradient descent.
Assistant: Gradient descent is an optimization algorithm...
```
تفاوت فقط در قالب‌بندی (Formatting) نیست. پس‌آموزش (Post-training) به مدل می‌آموزد:

- چه چیزی یک پاسخ مفید (Helpful Answer) محسوب می‌شود
- چگونه از دستورالعمل‌های وظیفه (Task Instructions) پیروی کند
- چگونه نقش‌های گفتگو (Dialogue Roles) را حفظ کند
- چگونه از ادامه‌های نامرتبط (Irrelevant Continuations) پرهیز کند
- چگونه پاسخ‌هایی را ترجیح دهد که انسان‌ها یا مدل‌های ترجیح (Preference Models) رتبهٔ بالاتری به آن‌ها می‌دهند
- چگونه میان مفیدبودن (Helpfulness)، صداقت (Honesty)، بی‌ضرر بودن (Harmlessness)، و محدودیت‌های دامنه (Domain Constraints) توازن برقرار کند

به همین دلیل است که پس‌آموزش (Post-training) برای کیفیت محصول در مدل‌های زبانی بزرگ مدرن (Modern LLM Product Quality) نقشی محوری دارد.

---

## 2. تنظیم دقیق نظارت‌شده (Supervised Fine-Tuning)

تنظیم دقیق نظارت‌شده (Supervised Fine-Tuning) مدل را بر روی نمایش‌هایی (Demonstrations) از رفتار مطلوب آموزش می‌دهد. هر مثال معمولاً شامل یک دستورالعمل (Instruction)، زمینهٔ اختیاری (Optional Context)، و یک پاسخ هدف (Target Response) است.

یک مثال حداقلی از SFT به‌صورت زیر است:

```json
{
  "instruction": "Summarize the following paragraph.",
  "input": "Large language models are trained on broad text corpora...",
  "output": "LLMs are trained on large text datasets to learn general language patterns."
}
```
محتوای تنظیم با دستورالعمل (Instruction-Tuning Material)، دستورالعمل (Instruction) را به‌عنوان فرمانی تعریف می‌کند که می‌خواهیم مدل از آن پیروی کند، و خروجی (Output) را به‌عنوان پاسخ مطلوبی که مدل باید تولید کند.

برای مدل‌های زبانی علّی (Causal Language Models)، SFT همچنان پیش‌بینی توکن بعدی (Next-Token Prediction) است. تفاوت در توزیع داده (Data Distribution) است. به‌جای متن دلخواه وب، مدل مثال‌های ساخت‌یافته‌ای از رفتاری را می‌بیند که ما می‌خواهیم.

تابع هدف استاندارد SFT به‌صورت زیر است:

$$ \mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t \in \mathcal{A}} \log \pi_{\theta}(y_t \mid x, y_{<t}) $$

که در آن:

- $$x$$ پرامپت (Prompt) یا زمینهٔ دستورالعمل (Instruction Context) است
- $$y$$ پاسخ هدف (Target Answer) است
- $$\mathcal{A}$$ مجموعهٔ توکن‌های پاسخ (Answer Tokens) است که برای محاسبهٔ زیان (Loss Computation) استفاده می‌شوند
- $$\pi_{\theta}$$ سیاست مدل زبانی قابل‌آموزش (Trainable Language Model Policy) است

جزئیات کلیدی این است که SFT به‌طور کلی نباید مدل را برای پیش‌بینی پرامپت (Prompt) آموزش دهد. این روش باید مدل را برای پیش‌بینی پاسخ دستیار (Assistant Response) آموزش دهد.

---

## 3. ماسک‌گذاری زیان (Loss Masking)

ماسک‌گذاری زیان (Loss Masking) یکی از مهم‌ترین جزئیات پیاده‌سازی در SFT است.

در تنظیم با دستورالعمل (Instruction Tuning)، دنبالهٔ ورودی اغلب هم شامل پرامپت کاربر (User Prompt) و هم پاسخ دستیار (Assistant Answer) است:

### Instruction:
Explain overfitting.

### Response:
Overfitting happens when a model performs well on training data but poorly on unseen data.

در طول آموزش، کل دنباله به مدل زبانی علّی (Causal LM) داده می‌شود. اما زیان (Loss) معمولاً باید فقط روی توکن‌های پاسخ (Response Tokens) اعمال شود. محتوای دوره این را مستقیماً بیان می‌کند: «در طول آموزش، ما معمولاً فقط به پیش‌بینی پاسخ علاقه‌مند هستیم، بنابراین زیان را برای پرامپت ماسک می‌کنیم.»

به‌صورت مفهومی:

```text
tokens:      [prompt tokens................][response tokens..............]
loss mask:   0 0 0 0 0 0 0 0 0 0 0 0 0     1 1 1 1 1 1 1 1 1 1 1
```
بدون ماسک‌گذاری زیان (Loss Masking)، مدل ظرفیت خود را صرف یادگیری بازتولید پرامپت‌های کاربر و پیشوندهای قالب‌بندی (Formatting Prefixes) می‌کند. با ماسک‌گذاری، گرادیان‌ها روی رفتار دستیار (Assistant Behavior) متمرکز می‌شوند.

در پیش‌پردازش به‌سبک Hugging Face، پرامپت قالب‌بندی‌شده و پاسخ در یک دنبالهٔ واحد توکنایز می‌شوند. مثال‌های دوره شامل تابعی هستند که `prompt_and_response` توکنایزشده را همراه با برش (Truncation) برمی‌گرداند، و همچنین به یک الگوی `f` برای ماسک‌گذاری بخش مربوط فقط به پرامپت اشاره می‌کنند.

یک نسخهٔ ساده‌شده به‌شکل زیر است:

```python
def format_example(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    response = example["output"]

    full_text = prompt + response
    prompt_ids = tokenizer(prompt).input_ids
    full_ids = tokenizer(full_text).input_ids

    labels = full_ids.copy()
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

    return {
        "input_ids": full_ids,
        "labels": labels,
    }
```
مقدار `-100` معمولاً توسط زیان آنتروپی متقاطع (Cross-Entropy Loss) در PyTorch نادیده گرفته می‌شود.

---

## 4. قالب‌های چت (Chat Templates)

مدل‌های چت (Chat Models) به قالب‌بندی سازگار (Consistent Formatting) نیاز دارند. یک قالب چت (Chat Template) نقش‌ها (Roles) و پیام‌ها (Messages) را به قالب متنی دقیقی که مدل انتظار دارد سریال‌سازی (Serialize) می‌کند.

یک گفتگو مانند این:

```json
[
  {"role": "system", "content": "You are a concise assistant."},
  {"role": "user", "content": "What is LoRA?"},
  {"role": "assistant", "content": "LoRA is a parameter-efficient fine-tuning method..."}
]
```
ممکن است به شکل زیر تبدیل شود:

```text
<|system|>
You are a concise assistant.
<|user|>
What is LoRA?
<|assistant|>
LoRA is a parameter-efficient fine-tuning method...
```
قالب دقیق، وابسته به مدل (Model-Specific) است. استفاده از قالب اشتباه (Wrong Template) می‌تواند عملکرد را کاهش دهد، زیرا مدل رفتاری را یاد گرفته است که به توکن‌های نقش مشخص (Particular Role Tokens)، جداکننده‌ها (Delimiters)، و پیشوندهای دستیار (Assistant Prefixes) وابسته است.

بنابراین، دادهٔ خوب برای پس‌آموزش (Good Post-Training Data) باید موارد زیر را استاندارد کند:

- پیام‌های سیستم (System Messages)
- پیام‌های کاربر (User Messages)
- پیام‌های دستیار (Assistant Messages)
- فراخوانی‌های ابزار (Tool Calls)، در صورت وجود
- سبک امتناع (Refusal Style)
- قالب‌بندی زمینهٔ چندنوبتی (Multi-Turn Context Formatting)
- رفتار پایان دنباله (End-of-Sequence Behavior)

برای SFT چت (Chat SFT)، قالب‌بندی داده بخشی از هدف آموزشی (Training Objective) است، نه صرفاً پیش‌پردازش (Preprocessing).

---

## 5. ساخت دادهٔ SFT (Building SFT Data)

دادهٔ SFT باید رفتاری را بازنمایی کند که در زمان استنتاج (Inference Time) انتظار می‌رود.

منابع رایج داده شامل موارد زیر هستند:

- جفت‌های دستورالعمل-پاسخ نوشته‌شده توسط انسان (Human-Written Instruction-Response Pairs)
- نمایش‌های تخصصی (Expert Demonstrations)
- دستورالعمل‌های مصنوعی تولیدشده توسط مدل‌های زبانی بزرگ قوی‌تر (Synthetic Instructions Generated by Stronger LLMs)
- مجموعه‌داده‌های وظیفه‌ای تبدیل‌شده به قالب دستورالعمل (Task Datasets Converted into Instruction Format)
- مثال‌های خاص دامنه از جریان‌های کاری تولید (Domain-Specific Examples from Production Workflows)
- ردپاهای استفاده از ابزار و مثال‌های استدلال ساخت‌یافته (Tool-Use Traces and Structured Reasoning Examples)

دادهٔ SFT باکیفیت باید ویژگی‌های زیر را داشته باشد:

- درست (Correct)
- پیرو دستورالعمل (Instruction-Following)
- متنوع (Diverse)
- سازگار از نظر قالب (Format-Consistent)
- عاری از مثال‌های تکراری (Free of Duplicated Examples)
- هم‌راستا با سیاست استقرار (Aligned with Deployment Policy)
- نزدیک به پرسش‌های مورد انتظار کاربران (Close to Expected User Queries)

کیفیت دادهٔ SFT اغلب از کمیت خام (Raw Quantity) مهم‌تر است. یک مجموعه‌دادهٔ کوچک از مثال‌های دقیق و نماینده (Precise, Representative Examples) می‌تواند از یک مجموعه‌دادهٔ بسیار بزرگ اما نویزی (Noisy Dataset) بهتر عمل کند.

یک رکورد عملی دادهٔ SFT معمولاً شامل موارد زیر است:

```json
{
  "messages": [
    {"role": "system", "content": "You are a technical assistant."},
    {"role": "user", "content": "Explain KL regularization in RLHF."},
    {"role": "assistant", "content": "KL regularization penalizes the policy when it moves too far from a reference model..."}
  ],
  "metadata": {
    "domain": "llm_training",
    "difficulty": "advanced",
    "source": "expert_written"
  }
}
```
فراداده (Metadata) همیشه در ورودی مدل گنجانده نمی‌شود، اما برای پالایش (Filtering)، تقسیم‌بندی ارزیابی (Evaluation Splits)، طراحی برنامهٔ درسی (Curriculum Design)، و ممیزی‌پذیری (Auditability) ارزشمند است.

---

## 6. تنظیم دقیق کامل و PEFT (Full Fine-Tuning and PEFT)

تنظیم دقیق کامل (Full Fine-Tuning) همهٔ پارامترهای مدل را به‌روزرسانی می‌کند. این کار می‌تواند سازگاری قدرتمندی ایجاد کند، اما به حافظهٔ قابل‌توجهی برای موارد زیر نیاز دارد:

- وزن‌های مدل (Model Weights)
- گرادیان‌ها (Gradients)
- وضعیت‌های بهینه‌ساز (Optimizer States)
- فعال‌سازی‌ها (Activations)
- ارتباطات آموزش توزیع‌شده (Distributed Training Communication)

برای مدل‌های زبانی بزرگ (Large LLMs)، تنظیم دقیق کامل می‌تواند پرهزینه یا غیرعملی باشد.

تنظیم دقیق کارآمد از نظر پارامتر (Parameter-Efficient Fine-Tuning) فقط تعداد کمی از پارامترهای اضافی یا انتخابی را به‌روزرسانی می‌کند. رایج‌ترین روش، LoRA است.

LoRA ماتریس وزن اصلی $W$ را منجمد (Freeze) می‌کند و یک به‌روزرسانی کم‌رتبه (Low-Rank Update) را یاد می‌گیرد:

$$ W' = W + \Delta W $$

$$ \Delta W = BA $$

که در آن:

- $$A \in \mathbb{R}^{r \times d}$$
- $$B \in \mathbb{R}^{k \times r}$$
- $$r$$ رتبهٔ LoRA (LoRA Rank) است
- $$r \ll \min(d, k)$$

به‌جای به‌روزرسانی $$W$$، آموزش فقط $$A$$ و $$B$$ را به‌روزرسانی می‌کند. این موضوع تعداد پارامترهای قابل‌آموزش (Trainable Parameters) و حافظهٔ بهینه‌ساز (Optimizer Memory) را کاهش می‌دهد.

محتوای تنظیم با دستورالعمل Hugging Face به‌طور صریح از `peft` برای تبدیل یک مدل پایه به مدل LoRA استفاده می‌کند.

PEFT زمانی مفید است که:

- حافظهٔ GPU محدود باشد
- به چندین آداپتور دامنه (Domain Adapters) نیاز باشد
- آموزش باید به‌طور مکرر تکرار شود
- مدل پایه باید بدون تغییر باقی بماند
- ادغام یا مسیریابی آداپتورها (Adapter Merging or Routing) مطلوب باشد

ابرپارامترهای رایج LoRA عبارت‌اند از:

- رتبه $$r$$
- ضریب مقیاس $$\alpha$$ (Scaling Factor)
- دراپ‌اوت (Dropout)
- ماژول‌های هدف (Target Modules)، مانند فرافکنی‌های توجه (Attention Projections)
- نرخ یادگیری (Learning Rate)
- این‌که آیا امبدینگ‌ها (Embeddings) یا سرهای خروجی (Output Heads) آموزش داده شوند یا نه

LoRA به‌صورت خودکار از تنظیم دقیق کامل بهتر نیست. این یک موازنه (Trade-off) است. هزینه و حافظه را کاهش می‌دهد، اما ممکن است ظرفیت سازگاری (Adaptation Capacity) را برای جابه‌جایی‌های بزرگ توزیع (Large Distribution Shifts) محدود کند.

---

## 7. دادهٔ ترجیح (Preference Data)

SFT به مدل می‌آموزد نمایش‌ها (Demonstrations) را تقلید کند. یادگیری ترجیح (Preference Learning) به مدل می‌آموزد میان پاسخ‌ها انتخاب کند.

یک مثال ترجیح معمولاً شامل موارد زیر است:

```json
{
  "prompt": "Explain RLHF.",
  "chosen": "RLHF fine-tunes a model using human preference feedback...",
  "rejected": "RLHF is when a model learns by itself from the internet..."
}
```
سیگنال کلیدی، مقایسه‌ای (Comparative) است. مدل فقط یک پاسخ درست را نمی‌بیند. بلکه می‌بیند که برای یک پرامپت یکسان، یک پاسخ نسبت به پاسخ دیگر ترجیح داده شده است.

دادهٔ ترجیح می‌تواند کیفیت‌هایی را ثبت کند که تعریف آن‌ها با هدف‌های دقیق (Exact Targets) دشوار است:

- مفیدبودن (Helpfulness)
- واقعیت‌مندی (Factuality)
- بی‌ضرر بودن (Harmlessness)
- اختصار (Concision)
- کیفیت استدلال (Reasoning Quality)
- کیفیت امتناع (Refusal Quality)
- لحن (Tone)
- پایبندی به دستورالعمل (Instruction Adherence)

دادهٔ ترجیح به دو روش اصلی استفاده می‌شود:

1. آموزش یک مدل پاداش (Reward Model)، سپس بهینه‌سازی سیاست (Policy) با RL.
2. آموزش مستقیم سیاست با یک هدف ترجیح (Preference Objective) مانند DPO.

---

## 8. مدل‌سازی پاداش (Reward Modeling)

یک مدل پاداش (Reward Model) یک جفت پرامپت-پاسخ (Prompt-Response Pair) را به یک نمرهٔ اسکالر (Scalar Score) نگاشت می‌کند:

$$ r_{\phi}(x, y) \in \mathbb{R} $$

که در آن:

- $$x$$ پرامپت (Prompt) است
- $$y$$ پاسخ (Response) است
- $$\phi$$ پارامترهای مدل پاداش (Reward Model Parameters) هستند

مدل پاداش باید به پاسخ ترجیح‌داده‌شده (Preferred Response) نمره‌ای بالاتر از پاسخ ردشده (Rejected Response) اختصاص دهد. محتوای دوره بیان می‌کند: «ما می‌خواهیم یک مدل پاداش را طوری آموزش دهیم که نمره‌ای که به پاسخ ترجیح‌داده‌شده می‌دهد، از نمره‌ای که به پاسخ ردشده می‌دهد بالاتر باشد.»

برای دادهٔ ترجیح دوتایی $$(x, y_w, y_l)$$ که در آن $$y_w$$ پاسخ انتخاب‌شده (Chosen Response) و $$y_l$$ پاسخ ردشده (Rejected Response) است، یک زیان رایج به‌صورت زیر است:

$$ \mathcal{L}_{\text{RM}}(\phi) = -\log \sigma\!\left(r_{\phi}(x, y_w) - r_{\phi}(x, y_l)\right) $$

این تابع از مدل ترجیح برادلی-تری (Bradley-Terry Preference Model) پیروی می‌کند. یادداشت‌های دوره توضیح می‌دهند که مدل Bradley-Terry را می‌توان به‌صورت یک سیگموید (Sigmoid) اعمال‌شده بر اختلاف دو نمرهٔ پاداش نوشت، و این‌که زیان مدل پاداش یک زیان آنتروپی متقاطع (Cross-Entropy Loss) بین ترجیح پیش‌بینی‌شده و ترجیح واقعی است.

یک مدل پاداش، اوراکل حقیقت (Truth Oracle) نیست. این فقط یک تقریب یادگرفته‌شده از برچسب‌های ترجیح (Preference Labels) است. ضعف‌های آن ممکن است در طول بهینه‌سازی RL مورد سوءاستفاده قرار گیرد و باعث هک پاداش (Reward Hacking) شود.

---

## 9. RLHF

یادگیری تقویتی از بازخورد انسانی (Reinforcement Learning from Human Feedback) از دادهٔ ترجیح انسانی برای بهینه‌سازی سیاست (Policy) یک مدل زبانی استفاده می‌کند.

محتوای RLHF این تکنیک را به‌عنوان روشی برای واردکردن بازخورد انسانی به فرایند تنظیم دقیق مدل‌های زبانی بزرگ (LLMs) توصیف می‌کند.

یک خط لولهٔ استاندارد RLHF سه مرحله دارد:

1. آموزش یک مدل SFT روی نمایش‌ها (Demonstrations).
2. آموزش یک مدل پاداش روی مقایسه‌های ترجیح انسانی (Human Preference Comparisons).
3. بهینه‌سازی سیاست SFT در برابر مدل پاداش با استفاده از یادگیری تقویتی (Reinforcement Learning).

مدل پاداش برای جفت‌های پرامپت-پاسخ یک پاداش اسکالر (Scalar Reward) فراهم می‌کند. محتوای دوره بیان می‌کند که پس از آموزش، مدل پاداش می‌تواند برای هر جفت پرامپت-پاسخ یک پاداش اسکالر ارائه دهد.

هدف RLHF اغلب به‌شکل زیر نوشته می‌شود:

$$ \max_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_{\theta}(\cdot \mid x)} \left[ r_{\phi}(x, y) - \beta D_{\mathrm{KL}}\!\left( \pi_{\theta}(\cdot \mid x)\;\Vert\;\pi_{\text{ref}}(\cdot \mid x) \right) \right] $$

که در آن:

- $$\pi_{\theta}$$ سیاست قابل‌آموزش (Trainable Policy) است
- $$\pi_{\text{ref}}$$ سیاست مرجع منجمد (Frozen Reference Policy) است که اغلب همان مدل SFT است
- $$r_{\phi}$$ مدل پاداش (Reward Model) است
- $$\beta$$ شدت منظم‌سازی KL (KL Regularization) را کنترل می‌کند

ترم KL مانع از آن می‌شود که سیاست بیش‌ازحد از مدل مرجع فاصله بگیرد. بدون این قید، سیاست ممکن است مصنوعات مدل پاداش (Reward Model Artifacts) را exploit کند و متن‌های غیرطبیعی یا کم‌کیفیت تولید کند.

---

## 10. PPO برای مدل‌های زبانی بزرگ (PPO for LLMs)

بهینه‌سازی سیاست مجاورتی (Proximal Policy Optimization) یک الگوریتم گرادیان سیاست (Policy-Gradient Algorithm) است که معمولاً در RLHF استفاده می‌شود.

PPO با محدودکردن میزان فاصله‌گرفتن سیاست جدید از سیاست قدیمی در یک به‌روزرسانی واحد، به‌روزرسانی‌های سیاست را پایدار می‌کند. محتوای PPO این هدف را به‌عنوان نسخهٔ بریده‌شده (Clipped Version) از هدف گرادیان سیاست توصیف می‌کند.

نسبت احتمال (Probability Ratio) را تعریف می‌کنیم:

$$ \rho_t(\theta) = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} $$

هدف بریده‌شدهٔ PPO به‌صورت زیر است:

$$ \mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t\!\left[ \min\!\left( \rho_t(\theta) A_t,\; \operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right] $$

که در آن:

- $$A_t$$ برآورد مزیت (Advantage Estimate) است
- $$\epsilon$$ دامنهٔ برش (Clipping Range) را کنترل می‌کند
- $$\rho_t$$ اندازهٔ تغییر سیاست را اندازه‌گیری می‌کند

برای مدل‌های زبانی بزرگ:

- مدل سیاست (Policy Model) توکن‌ها را تولید می‌کند
- مدل پاداش (Reward Model) پاسخ‌های کامل‌شده را امتیازدهی می‌کند
- سر ارزش (Value Head) بازده موردانتظار (Expected Return) را تخمین می‌زند
- مدل مرجع (Reference Model) رفتار را از طریق جریمهٔ KL منظم می‌کند

محتوای آموزش PPO از مدلی با سر ارزش، یعنی `AutoModelForCausalLMWithValueHead` استفاده می‌کند که نیاز به تخمین ارزش در طول PPO را بازتاب می‌دهد.

همین محتوا همچنین بارگذاری هم‌زمان مدل و مدل مرجع را نشان می‌دهد.

`PPOTrainer`

PPO قدرتمند است، اما از نظر عملیاتی پیچیده است. این روش نیازمند مدیریت دقیق موارد زیر است:

- مقیاس‌بندی پاداش (Reward Scaling)
- ضریب KL (KL Coefficient)
- تولید رول‌اوت (Rollout Generation)
- اندازهٔ بچ (Batch Size)
- زیان ارزش (Value Loss)
- برآورد مزیت (Advantage Estimation)
- دمای نمونه‌گیری (Sampling Temperature)
- کیفیت مدل پاداش (Reward Model Quality)
- ناپایداری ناشی از رانش سیاست (Policy Drift)

این پیچیدگی یکی از دلایلی است که روش‌های ترجیح مستقیم مانند DPO به‌طور گسترده پذیرفته شدند.

---

## 11. بهینه‌سازی مستقیم ترجیح (Direct Preference Optimization)

بهینه‌سازی مستقیم ترجیح (Direct Preference Optimization) سیاست را مستقیماً از زوج‌های ترجیح (Preference Pairs) آموزش می‌دهد، بدون آن‌که یک مدل پاداش جداگانه آموزش داده شود و بدون اجرای یک حلقهٔ صریح یادگیری تقویتی (Explicit Reinforcement Learning Loop).

محتوای DPO این روش را به‌عنوان روشی برای تنظیم دقیق مدل‌های زبانی بزرگ با ترجیحات انسانی توصیف می‌کند که از آموزش یک مدل پاداش جداگانه اجتناب می‌کند و فرایند آموزش را ساده‌تر می‌سازد.

DPO از همان دادهٔ ترجیح شروع می‌کند:

```text
prompt x
chosen response y_w
rejected response y_l
```
زیان DPO به‌صورت زیر است:

$$ \mathcal{L}_{\text{DPO}}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\!\left( \beta \left[ \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right] \right) \right] $$

که در آن:

- $$y_w$$ پاسخ انتخاب‌شده (Chosen Response) است
- $$y_l$$ پاسخ ردشده (Rejected Response) است
- $$\pi_{\theta}$$ سیاست قابل‌آموزش (Trainable Policy) است
- $$\pi_{\text{ref}}$$ مدل مرجع منجمد (Frozen Reference Model) است
- $$\beta$$ شدت به‌روزرسانی ترجیح نسبت به مدل مرجع را کنترل می‌کند

محتوای دوره مسیر استخراج از سیاست بهینه (Optimal Policy) تا هدف DPO را برجسته می‌کند و به‌صورت جداگانه دربارهٔ استخراج راه‌حل سیاست بهینه بحث می‌کند.

پارامتر $$\beta$$ مهم است. محتوای DPO از Hugging Face بیان می‌کند که بتا (Beta) شدت منظم‌سازی واگرایی KL (KL-Divergence Regularization) را در زیان DPO کنترل می‌کند.

$$\beta$$ بزرگ‌تر باعث می‌شود مدل با شدت بیشتری به تفاوت‌های ترجیح واکنش نشان دهد. $$\beta$$ کوچک‌تر مدل را به سیاست مرجع نزدیک‌تر نگه می‌دارد.

DPO جذاب است زیرا از چندین دشواری RLHF/PPO اجتناب می‌کند:

- به مدل پاداش جداگانه نیاز ندارد
- به حلقهٔ رول‌اوت برخط (Online Rollout Loop) نیاز ندارد
- به مدل ارزش (Value Model) نیاز ندارد
- آموزش آن شبیه یادگیری نظارت‌شده (Supervised Learning) است
- پیاده‌سازی آن ساده‌تر و پایدارتر است

بااین‌حال، DPO همچنان به‌شدت به کیفیت دادهٔ ترجیح (Preference Data Quality) وابسته است. اگر جفت‌های انتخاب‌شده/ردشده نویزی، سوگیرانه یا با تفکیک ضعیف باشند، DPO می‌تواند ترجیحات اشتباه را یاد بگیرد.

---

## 12. مقایسهٔ SFT، RLHF، PPO و DPO (Comparing SFT, RLHF, PPO, and DPO)

هر روش پس‌آموزش (Post-Training Method) مسئلهٔ متفاوتی را حل می‌کند.

| روش (Method) | سیگنال آموزشی (Training Signal) | کاربرد اصلی (Main Use) | نقطهٔ قوت (Strength) | محدودیت (Limitation) |
|---|---|---|---|---|
| SFT | پاسخ نمایشی (Demonstration Response) | آموزش قالب رفتار مطلوب (Teach Desired Behavior Format) | ساده و پایدار (Simple and Stable) | داده را تقلید می‌کند، ترجیحات را مستقیماً بهینه نمی‌کند (Imitates Data, Does Not Directly Optimize Preferences) |
| مدل‌سازی پاداش (Reward Modeling) | پاسخ انتخاب‌شده در برابر پاسخ ردشده (Chosen vs. Rejected Response) | یادگیری نمرهٔ ترجیح اسکالر (Learn Scalar Preference Score) | یادگیری ترجیح را از بهینه‌سازی سیاست جدا می‌کند (Separates Preference Learning from Policy Optimization) | مدل پاداش قابل exploit شدن است (Reward Model Can Be Exploited) |
| RLHF | مدل پاداش + RL (Reward Model + RL) | بهینه‌سازی سیاست برای ترجیحات یادگرفته‌شده (Optimize Policy for Learned Preferences) | سیگنال هم‌راستاسازی قوی (Strong Alignment Signal) | پیچیده و ناپایدار (Complex and Unstable) |
| PPO | RL مبتنی بر گرادیان سیاست (Policy-Gradient RL) | بهینه‌ساز استاندارد RLHF (Standard RLHF Optimizer) | به‌روزرسانی‌های سیاست را با برش و KL کنترل می‌کند (Controls Policy Updates with Clipping and KL) | از نظر عملیاتی پرهزینه است (Operationally Expensive) |
| DPO | پاسخ انتخاب‌شده در برابر پاسخ ردشده (Chosen vs. Rejected Response) | تنظیم مستقیم ترجیح (Direct Preference Tuning) | ساده‌تر از PPO است و به مدل پاداش نیاز ندارد (Simpler than PPO, No Reward Model Required) | به کیفیت دادهٔ ترجیح حساس است (Sensitive to Preference Data Quality) |

یک دستورالعمل عملی رایج:

```text
base model
 -> SFT
 -> preference tuning with DPO or PPO
 -> evaluation
 -> safety and domain-specific refinement
```
برای بسیاری از تیم‌ها، DPO روش پیش‌فرض تنظیم ترجیح (Preference-Tuning Method) است، زیرا اجرای آن از PPO آسان‌تر است. PPO همچنان زمانی مفید است که بهینه‌سازی برخط پاداش (Online Reward Optimization)، تعامل با محیط (Environment Interaction)، یا تنظیمات پیچیده‌تر RL موردنیاز باشد.

---

## 13. جریان کاری عملی پس‌آموزش (Practical Post-Training Workflow)

یک جریان کاری قوی برای پس‌آموزش (Robust Post-Training Workflow) معمولاً این توالی را دنبال می‌کند:

1. انتخاب یک مدل پایه (Base Model) متناسب با دامنهٔ هدف و بودجه.
2. آماده‌سازی دادهٔ دستورالعمل با قالب‌بندی سازگار (Consistent Formatting).
3. اعمال SFT با ماسک‌گذاری زیان صحیح (Correct Loss Masking).
4. ارزیابی رفتار پیروی از دستورالعمل (Instruction-Following Behavior).
5. جمع‌آوری یا تولید زوج‌های ترجیح (Preference Pairs).
6. آموزش با DPO یا آموزش یک مدل پاداش برای RLHF.
7. پایش پس‌رفت‌ها (Regressions)، ایمنی (Safety)، و عملکرد دامنه (Domain Performance).
8. اجرای ارزیابی انسانی یا ارزیابی با کمک مدل (Model-Assisted Evaluation).
9. بسته‌بندی مدل نهایی با توکنایزر (Tokenizer) و قالب چت (Chat Template) صحیح.
10. مستندسازی دادهٔ آموزشی (Training Data)، ابرپارامترها (Hyperparameters)، و محدودیت‌های شناخته‌شده (Known Limitations).

برای جریان‌های کاری به‌سبک Hugging Face/TRL، محتوای دوره به موارد زیر اشاره می‌کند:

- `RewardTrainer` برای آموزش مدل پاداش
- `PPOTrainer` برای PPO
- ابزارهای Hugging Face برای DPO
- پیکربندی PEFT برای تنظیم دقیق کارآمد از نظر حافظه (Memory-Efficient Fine-Tuning)

---

## 14. حالت‌های شکست (Failure Modes)

پس‌آموزش (Post-training) می‌تواند رفتار مدل را به‌طور قابل‌توجهی بهبود دهد، اما حالت‌های شکست (Failure Modes) نیز ایجاد می‌کند.

### فراموشی فاجعه‌بار (Catastrophic Forgetting)

مدل ممکن است قابلیت‌هایی را که در طول پیش‌آموزش (Pretraining) یا تنظیم دقیق قبلی یاد گرفته است از دست بدهد. این وضعیت زمانی محتمل‌تر است که دادهٔ تنظیم دقیق محدود، تکراری، یا بیش‌ازحد خاص دامنه باشد.

راهکارهای کاهش (Mitigations) شامل موارد زیر هستند:

- نرخ‌های یادگیری پایین‌تر (Lower Learning Rates)
- ایپاک‌های کمتر (Fewer Epochs)
- دادهٔ آموزشی چنددامنه‌ای ترکیبی (Mixed-Domain Training Data)
- ارزیابی منظم روی بنچمارک‌های عمومی (Regular Evaluation on General Benchmarks)
- منظم‌سازی KL (KL Regularization)
- استفاده از PEFT به‌جای تنظیم دقیق کامل
- بازپخش داده از توزیع‌های قبلی (Replay Data from Earlier Distributions)

### بیش‌بهینه‌سازی (Over-Optimization)

در RLHF، سیاست ممکن است نقص‌های مدل پاداش را exploit کند. ممکن است پاسخ‌هایی یاد بگیرد که امتیاز بالایی می‌گیرند اما پرحرف، طفره‌آمیز، کلیشه‌ای (Formulaic)، یا از نظر واقعی ضعیف هستند.

راهکارهای کاهش شامل موارد زیر هستند:

- ارزیابی مدل پاداش (Reward Model Evaluation)
- کنترل KL
- بررسی‌های موردی انسانی (Human Spot Checks)
- دادهٔ ترجیح خصمانه (Adversarial Preference Data)
- بازآموزی مدل پاداش (Reward Model Retraining)
- تنظیمات محافظه‌کارانهٔ PPO (Conservative PPO Settings)

### بیش‌برازش به قالب (Format Overfitting)

مدل ممکن است بیش‌ازحد به یک قالب پرامپت (Prompt Template) یا سبک پاسخ وابسته شود.

راهکارهای کاهش شامل موارد زیر هستند:

- تنوع قالب (Template Diversity)
- دادهٔ چندنوبتی واقع‌گرایانه (Realistic Multi-Turn Data)
- ارزیابی در قالب‌های مختلف پرامپت (Evaluation Across Prompt Formats)
- جداسازی روشن نقش‌های سیستم، کاربر، و دستیار (Clear Separation Between System, User, and Assistant Roles)

### نویز در دادهٔ ترجیح (Preference Data Noise)

برچسب‌های ترجیح ممکن است ناسازگار، ذهنی، یا کم‌کیفیت باشند. DPO و مدل‌سازی پاداش هر دو نسبت به این موضوع حساس هستند.

راهکارهای کاهش شامل موارد زیر هستند:

- راهنماهای برچسب‌گذار (Annotator Guidelines)
- فیلتر کیفیت جفت‌ها (Pair Quality Filtering)
- اندازه‌گیری توافق (Agreement Measurement)
- استخراج نمونه‌های منفی سخت (Hard-Negative Mining)
- بازبینی توسط متخصص دامنه (Domain Expert Review)

### پس‌رفت ایمنی (Safety Regression)

تنظیم دقیق روی داده‌های دامنه‌ای یا کاربری می‌تواند رفتار ایمنی (Safety Behavior) را تضعیف کند، اگر مثال‌های امتناع (Refusal)، عدم‌قطعیت (Uncertainty)، و سیاست (Policy) در داده حضور نداشته باشند.

راهکارهای کاهش شامل موارد زیر هستند:

- مجموعه‌های ارزیابی ایمنی (Safety Evaluation Sets)
- مثال‌های امتناع (Refusal Examples)
- پرامپت‌های رد-تیم (Red-Team Prompts)
- زوج‌های ترجیح هم‌راستا با سیاست (Policy-Aligned Preference Pairs)
- ممیزی‌های پس‌آموزش (Post-Training Audits)

---

## 15. نکات کلیدی (Key Takeaways)

پس‌آموزش (Post-training) یک مدل پایه (Base Model) را به یک مدل قابل‌استفاده تبدیل می‌کند. SFT به مدل می‌آموزد پاسخ‌های مطلوب را تقلید کند. ماسک‌گذاری زیان (Loss Masking) تضمین می‌کند که آموزش روی خروجی‌های دستیار متمرکز باشد، نه پرامپت‌های کاربر. روش‌های PEFT مانند LoRA تنظیم دقیق مدل‌های بزرگ را ارزان‌تر و ماژولارتر می‌کنند.

یادگیری ترجیح (Preference Learning) فراتر از تقلید (Imitation) می‌رود. مدل‌سازی پاداش (Reward Modeling) نمره‌های ترجیح اسکالر را از جفت‌های انتخاب‌شده/ردشده یاد می‌گیرد. RLHF از این نمره‌ها برای بهینه‌سازی یک سیاست استفاده می‌کند، اغلب با PPO و منظم‌سازی KL در برابر یک مدل مرجع. DPO تنظیم ترجیح را با بهینه‌سازی مستقیم سیاست روی زوج‌های ترجیح، بدون مدل پاداش جداگانه یا حلقهٔ صریح RL، ساده می‌کند.

چالش اصلی مهندسی فقط انتخاب الگوریتم نیست. مسئله، کنترل کیفیت داده، قالب‌بندی، ارزیابی، پایداری، و پس‌رفت‌ها در سراسر خط لولهٔ کامل پس‌آموزش (Full Post-Training Pipeline) است.

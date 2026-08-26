---
id: transformers
title: معماری ترنسفورمرها
sidebar_label: معماری ترنسفورمرها
sidebar_position: 5
---

<div className="chapter-hero">

![ فصل ۴ — معماری ترنسفورمرزها](/img/chapters/transformer-architecture.png)

</div>

[قبلی: توکن‌سازی](./03-tokenization.md) |
[فهرست مطالب](./index.md) |
[بعدی: پیش‌آموزش](./05-pretraining.md)

---


## اهداف یادگیری (Learning Objectives)

تا پایان این فصل، باید بتوانید:

- توضیح دهید چرا ترنسفورمر (Transformer) در مدل‌سازی زبان در مقیاس بزرگ، جایگزین معماری‌های بازگشتی شد.
- اجزای اصلی یک بلوک ترنسفورمر (Transformer block) را توصیف کنید.
- بردارهای تعبیه‌ی توکن (Token Embeddings)، اطلاعات مکانی/موقعیتی (Positional Information)، خودتوجهی (Self-Attention)، شبکه‌های پیش‌خور (Feed-Forward Networks)، نرمال‌سازی (Normalization) و اتصال‌های باقیمانده (Residual Connections) را درک کنید.
- معادلات اصلی توجه ضرب داخلی مقیاس‌شده (Scaled Dot-Product Attention) را استخراج کنید.
- توجه چندسری (Multi-Head Attention) و دلیل مفید بودن چندین سر توجه را توضیح دهید.
- تفاوت میان معماری‌های ترنسفورمر فقط-رمزگذار (Encoder-only)، فقط-رمزگشا (Decoder-only) و رمزگذار-رمزگشا (Encoder-decoder) را مشخص کنید.
- توضیح دهید چرا LLMهای مدرن معمولاً از ترنسفورمرهای علّی فقط-رمزگشا (Decoder-only Causal Transformers) استفاده می‌کنند.
- ماسک‌گذاری علّی (Causal Masking) و این‌که چگونه تولید خودبازگشتی (Autoregressive Generation) را ممکن می‌کند، درک کنید.
- واریانت‌های معماری رایج در LLMهای مدرن، از جمله Pre-LN، RMSNorm، SwiGLU، RoPE، Grouped-Query Attention و Multi-Query Attention را توصیف کنید.
- تحلیل کنید که معماری ترنسفورمر چگونه بر محاسبه (Compute)، حافظه (Memory)، طول زمینه (Context Length) و کارایی استنتاج (Inference Efficiency) اثر می‌گذارد.
- خطاهای رایج پیاده‌سازی در مدل‌های ترنسفورمر را شناسایی کنید.

---

## ۴.۱ چرا ترنسفورمرها مهم‌اند؟

ترنسفورمر، معماری غالب پشت مدل‌های زبانی بزرگ مدرن است.

پیش از ترنسفورمرها، بسیاری از مدل‌های دنباله‌ای از شبکه‌های عصبی بازگشتی مانند RNN، LSTM یا GRU استفاده می‌کردند. این مدل‌ها متن را به‌صورت توکن‌به‌توکن و در ترتیب دنباله پردازش می‌کردند. این ویژگی، یادگیری وابستگی‌های بلندبرد (Long-range Dependencies) را دشوار می‌کرد و همچنین میزان موازی‌سازی (Parallelism) در زمان آموزش را محدود می‌ساخت.

ترنسفورمر این وضعیت را با استفاده از **توجه** (Attention) به‌عنوان سازوکار اصلی برای آمیختن اطلاعات میان توکن‌ها تغییر داد، در حالی که هم‌زمان اجازه می‌داد موقعیت‌های مختلف دنباله در زمان آموزش به‌صورت موازی پردازش شوند.

به‌جای آن‌که توکن‌ها الزاماً یکی‌یکی و به‌صورت سخت‌گیرانه پردازش شوند، یک ترنسفورمر می‌تواند در زمان آموزش همه‌ی توکن‌های یک دنباله را به‌صورت موازی پردازش کند، در حالی که بازنمایی هر توکن همچنان قادر است به توکن‌های دیگر توجه کند.

این تغییر، امکان مقیاس‌دادن مدل‌های زبانی را به این ابعاد فراهم کرد:

- داده‌مجموعه‌های بزرگ‌تر
- دنباله‌های طولانی‌تر
- پارامترهای بیشتر
- سخت‌افزارهای موازی‌تر برای آموزش
- استدلال بهتر در زمینه‌های طولانی
- انتقال یادگیری (Transfer Learning) بهتر
- رفتار همه‌منظوره‌تر (General-purpose)

ترنسفورمر اولیه در مقاله‌ی زیر معرفی شد:

> Vaswani et al., *Attention Is All You Need*, 2017.

LLMهای مدرن با ترنسفورمر اولیه کاملاً یکسان نیستند، اما به‌طور مستقیم از آن مشتق شده‌اند.

---

## ۴.۲ نمای سطح‌بالای ترنسفورمر

یک مدل زبانی ترنسفورمر، دنباله‌ای از شناسه‌های توکن (Token IDs) را می‌گیرد و روی توکن بعدی، یک توزیع احتمال تولید می‌کند.

یک ترنسفورمر ساده‌ی فقط-رمزگشا (Decoder-only) به‌صورت زیر است:

```text
Token IDs
   |
   v
Token Embeddings + Positional Information
   |
   v
Transformer Block 1
   |
   v
Transformer Block 2
   |
   v
...
   |
   v
Transformer Block N
   |
   v
Final Normalization
   |
   v
Output Projection
   |
   v
Logits over Vocabulary
   |
   v
Softmax
   |
   v
Next-token probabilities
```
هر بلوک ترنسفورمر معمولاً شامل این ساختار است:

```text
Input
  |
  +--> Normalization
  |       |
  |       v
  |   Self-Attention
  |       |
  +<------+
  |
  +--> Normalization
  |       |
  |       v
  |   Feed-Forward Network
  |       |
  +<------+
  |
Output
```
دو زیرلایه‌ی اصلی عبارت‌اند از:

- **خودتوجهی (Self-attention)**
- **شبکه‌ی پیش‌خور (Feed-forward network)** که گاهی **MLP** یا **FFN** نیز نامیده می‌شود.

این اجزا از طریق عناصر زیر به هم متصل می‌شوند:

- اتصال‌های باقیمانده (Residual Connections)
- لایه‌های نرمال‌سازی (Normalization Layers)
- اطلاعات موقعیتی (Positional Information)
- Dropout در برخی تنظیمات آموزشی

---

## ۴.۳ از توکن‌ها به بردارها

همان‌طور که در فصل ۳ دیدیم، متن ابتدا به شناسه‌های توکن تبدیل می‌شود.

اگر دنباله‌ای به‌صورت زیر داشته باشیم:

$$
x_1, x_2, \ldots, x_T
$$

که در آن هر $x_t$ یک شناسه‌ی توکن است، مدل هر شناسه‌ی توکن را با استفاده از یک ماتریس embedding به یک بردار چگال (Dense Vector) نگاشت می‌کند:

$$
E \in \mathbb{R}^{|V| \times d_{\text{model}}}
$$

embedding متناظر با توکن $x_t$ برابر است با:

$$
h_t^{(0)} = E[x_t]
$$

که در آن:

- $|V|$ اندازه‌ی واژگان (Vocabulary Size) است.
- $d_{\text{model}}$ بُعد نهان یا عرض مدل (Hidden Dimension / Model Width) است.
- $h_t^{(0)}$ بازنمایی اولیه‌ی توکن $t$ است.

برای کل دنباله داریم:

$$
H^{(0)} \in \mathbb{R}^{T \times d_{\text{model}}}
$$

که در آن:

- $T$ طول دنباله (Sequence Length) است.
- $d_{\text{model}}$ عرض مدل است.

ترنسفورمر این ماتریس را در طول چندین لایه، بارها به‌روزرسانی می‌کند.

---

## ۴.۴ چرا موقعیت اهمیت دارد؟

خودتوجهی به‌تنهایی نسبت به جایگشت (Permutation-equivariant) است؛ یعنی اگر اطلاعات موقعیتی وجود نداشته باشد، ترتیب توکن‌ها را تشخیص نمی‌دهد.

اگر این دو دنباله را به مدل بدهیم:

```text
The dog chased the cat.
The cat chased the dog.
```
معنای آن‌ها متفاوت است. اما بدون اطلاعات موقعیتی، attention عملاً مجموعه‌ای از بردارهای توکن را می‌بیند، نه یک دنباله‌ی مرتب.

بنابراین، ترنسفورمرها به روشی برای نمایش موقعیت (Position) نیاز دارند.

رویکردهای رایج شامل این‌ها هستند:

- Learned absolute positional embeddings
- Sinusoidal positional embeddings
- Relative positional bias
- Rotary positional embeddings یا RoPE
- ALiBi
- سایر روش‌های مخصوص زمینه‌های طولانی

در LLMهای مدرن فقط-رمزگشا، معمولاً از RoPE یا واریانت‌های نزدیک به آن استفاده می‌شود.

---

## ۴.۵ embeddingهای موقعیتی مطلق

ترنسفورمر اولیه از کدگذاری موقعیتی سینوسی (Sinusoidal Positional Encoding) استفاده می‌کرد. برخی مدل‌های بعدی از embeddingهای موقعیتی مطلقِ آموختنی (Learned Absolute Positional Embeddings) استفاده کردند.

در موقعیت‌های مطلقِ آموختنی، مدل یک ماتریس embedding موقعیت دارد:

$$
P \in \mathbb{R}^{T_{\max} \times d_{\text{model}}}
$$

بازنمایی ورودی به این صورت می‌شود:

$$
h_t^{(0)} = E[x_t] + P[t]
$$

که در آن $P[t]$ embedding موقعیتیِ موقعیت $t$ است.

### مزایا

- پیاده‌سازی ساده
- عملکرد مناسب در محدوده‌ی طول زمینه‌ای که مدل روی آن آموزش دیده
- درک شهودی و ساده

### معایب

- به‌صورت طبیعی به زمینه‌های طولانی‌تر تعمیم نمی‌یابد.
- به یک جدول موقعیتی با طول بیشینه‌ی ثابت نیاز دارد.
- در زمان packing و generation باید با شناسه‌های موقعیت با دقت رفتار شود.

بسیاری از مدل‌های قدیمی‌تر ترنسفورمر از embeddingهای موقعیتی مطلق استفاده می‌کردند، اما در LLMهای مدرن معمولاً از جایگزین‌های دیگر استفاده می‌شود.

---

## ۴.۶ کدگذاری موقعیتی سینوسی

ترنسفورمر اولیه از توابع سینوسی ثابت استفاده می‌کرد:

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

که در آن:

- $pos$ موقعیت توکن است.
- $i$ اندیس بُعد embedding را نشان می‌دهد.

کدگذاری‌های سینوسی به مدل اجازه می‌دهند که از طریق ترکیب توابع تناوبی، فاصله‌های نسبی را نمایش دهد.

امروزه این روش انتخاب پیش‌فرض بسیاری از LLMهای مدرن نیست، اما از نظر تاریخی بسیار مهم است.

---

## ۴.۷ embeddingهای موقعیتی چرخشی (RoPE)

**Rotary Positional Embeddings** یا **RoPE** اطلاعات موقعیتی را با چرخاندن (Rotating) بردارهای Query و Key در attention وارد می‌کنند.

به‌جای آن‌که یک بردار موقعیت به embedding توکن‌ها اضافه شود، در RoPE یک چرخش وابسته به موقعیت روی بردارهای query و key اعمال می‌شود.

در سطحی کلی:

```text
Token hidden state
   |
   v
Query and key projections
   |
   v
Apply position-dependent rotation
   |
   v
Attention score computation
```
RoPE محبوب است، زیرا:

- اطلاعات موقعیتی نسبی را به‌صورت طبیعی کد می‌کند.
- در LLMهای فقط-رمزگشا عملکرد خوبی دارد.
- با پیاده‌سازی‌های کارآمد attention سازگار است.
- از تکنیک‌های گسترش زمینه‌ی طولانی (Long-context Extension) پشتیبانی می‌کند.
- به جدول موقعیت مطلقِ آموختنی نیاز ندارد.

بسیاری از خانواده‌های مدرن LLM متن‌باز از RoPE استفاده می‌کنند.


---

## ۴.۸ بلوک ترنسفورمر

یک بلوک استاندارد فقط-رمزگشا در ترنسفورمر شامل این اجزاست:

```text
Input hidden states
|
v
Normalization
|
v
Causal self-attention
|
v
Residual addition
|
v
Normalization
|
v
Feed-forward network
|
v
Residual addition
|
v
Output hidden states
```
در قالب معادله، یک بلوک رایج Pre-LN به‌صورت زیر نوشته می‌شود:

$$
\tilde{H}^{(l)} = H^{(l)} + \text{Attention}(\text{Norm}(H^{(l)}))
$$

$$
H^{(l+1)} = \tilde{H}^{(l)} + \text{FFN}(\text{Norm}(\tilde{H}^{(l)}))
$$

که در آن:

- $H^{(l)}$ ورودی لایه‌ی $l$ است.
- $H^{(l+1)}$ خروجی لایه‌ی $l$ است.
- $\text{Norm}$ معمولاً LayerNorm یا RMSNorm است.
- $\text{FFN}$ شبکه‌ی پیش‌خور است.

---

## ۴.۹ شهود خودتوجهی

خودتوجهی به هر توکن اجازه می‌دهد به توکن‌های دیگر در همان دنباله نگاه کند.

مثال:

```text
The animal did not cross the street because it was tired.
```
واژه‌ی `it` ممکن است لازم باشد به `animal` ارجاع دهد.

خودتوجهی به بازنمایی `it` اجازه می‌دهد از اطلاعات توکن‌های قبلی استفاده کند.

در یک مدل زبانی علّی (Causal Language Model)، توکن $t$ فقط می‌تواند به موقعیت‌های زیر توجه کند:

$$
1, 2, \ldots, t
$$

یعنی در زمان آموزش یا generation، حق ندارد به توکن‌های آینده توجه کند.

این محدودیت، هدف خودبازگشتی (Autoregressive Objective) را حفظ می‌کند.

---

## ۴.۱۰ Query، Key و Value

خودتوجهی از سه projection آموختنی استفاده می‌کند:

- Query
- Key
- Value

اگر hidden stateها به‌صورت زیر باشند:

$$
H \in \mathbb{R}^{T \times d_{\text{model}}}
$$

مدل این کمیت‌ها را محاسبه می‌کند:

$$
Q = HW_Q
$$

$$
K = HW_K
$$

$$
V = HW_V
$$

که در آن:

$$
W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}
$$

هر توکن سه بردار تولید می‌کند:

- بردار query: این توکن به‌دنبال چه چیزی است.
- بردار key: این توکن برای تطبیق چه چیزی عرضه می‌کند.
- بردار value: اگر به این توکن توجه شود، چه اطلاعاتی منتقل می‌شود.

امتیازهای attention با مقایسه‌ی queryها و keyها محاسبه می‌شوند.

---

## ۴.۱۱ توجه ضرب داخلی مقیاس‌شده

عملیات اصلی attention به‌صورت زیر است:

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^\top}{\sqrt{d_k}} \right) V
$$

که در آن:

- $Q \in \mathbb{R}^{T \times d_k}$
- $K \in \mathbb{R}^{T \times d_k}$
- $V \in \mathbb{R}^{T \times d_v}$
- $d_k$ بُعد key/query است.

ماتریس

$$
QK^\top \in \mathbb{R}^{T \times T}
$$

امتیازهای attention جفتی میان توکن‌ها را در خود دارد.

عامل مقیاس

$$
\frac{1}{\sqrt{d_k}}
$$

باعث می‌شود با رشد بُعد، ضرب داخلی‌ها بیش از حد بزرگ نشوند.

بدون این مقیاس‌گذاری، softmax می‌تواند به ناحیه‌ی اشباع برود و گرادیان‌ها ضعیف شوند.

---

## ۴.۱۲ ماتریس attention

برای دنباله‌ای با طول $T$، attention یک ماتریس امتیاز $T \times T$ تولید می‌کند.

مثال برای $T = 4$:

```text
          key1   key2   key3   key4
query1      .      .      .      .
query2      .      .      .      .
query3      .      .      .      .
query4      .      .      .      .

هر سطر متناظر با یک توکن query و هر ستون متناظر با یک توکن key است.
```
پس از اعمال softmax، هر سطر به یک توزیع احتمال روی توکن‌ها تبدیل می‌شود.

خروجی برای توکن $i$ یک مجموع وزن‌دار از بردارهای value است:

$$
o_i = \sum_{j=1}^{T} a_{ij} v_j
$$

که در آن:

- $a_{ij}$ وزن attention از توکن $i$ به توکن $j$ است.
- $v_j$ بردار value مربوط به توکن $j$ است.

---

## ۴.۱۳ ماسک‌گذاری علّی

در مدل‌سازی زبان خودبازگشتی، توکن $t$ نباید به توکن‌های آینده توجه کند.

مدل این کمیت را پیش‌بینی می‌کند:

$$
P(x_t \mid x_{\lt t})
$$

بنابراین، هنگام محاسبه‌ی attention برای موقعیت $t$، همه‌ی موقعیت‌های بزرگ‌تر از $t$ باید mask شوند.

برای دنباله‌ای با طول 4، ماسک علّی این الگو را مجاز می‌کند:

```text
          key1   key2   key3   key4
query1      ✓      x      x      x
query2      ✓      ✓      x      x
query3      ✓      ✓      ✓      x
query4      ✓      ✓      ✓      ✓
```
امتیازهای attention ماسک‌شده، پیش از softmax به یک مقدار بسیار منفی تنظیم می‌شوند:

$$
-\infty
$$

در نتیجه احتمال softmax آن‌ها تقریباً صفر می‌شود.

این کار از نشت اطلاعات (Information Leakage) از آینده جلوگیری می‌کند.

---

## ۴.۱۴ چرا ماسک‌گذاری علّی مهم است؟

فرض کنید دنباله‌ی آموزشی چنین باشد:

```text
The capital of France is Paris.
```
هنگام پیش‌بینی `Paris`، مدل می‌تواند به این بخش توجه کند:

```text
The capital of France is
```
اما نباید به خود `Paris` یا هیچ توکن بعدی نگاه کند.

اگر ماسک‌گذاری علّی به‌درستی پیاده‌سازی نشود، مدل ممکن است در زمان آموزش توکن‌های آینده را ببیند. این موضوع loss آموزشی را به‌صورت مصنوعی پایین می‌آورد، اما مدل در زمان generation شکست می‌خورد.

این یکی از جدی‌ترین باگ‌های پیاده‌سازی در آموزش مدل‌های زبانی است.

---

## ۴.۱۵ توجه چندسری

به‌جای آن‌که attention فقط یک‌بار اجرا شود، ترنسفورمرها از چندین سر attention استفاده می‌کنند.

هر سر attention projectionهای query، key و value مخصوص به خود را دارد.

برای سر $h$:

$$
Q_h = HW_Q^{(h)}
$$

$$
K_h = HW_K^{(h)}
$$

$$
V_h = HW_V^{(h)}
$$

$$
O_h = \text{Attention}(Q_h, K_h, V_h)
$$

خروجی همه‌ی سرها با هم الحاق (Concatenate) می‌شوند:

$$
O = \text{Concat}(O_1, O_2, \ldots, O_n)
$$

سپس دوباره به بُعد مدل projection می‌شوند:

$$
\text{MHA}(H) = OW_O
$$

که در آن:

$$
W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$

توجه چندسری به مدل اجازه می‌دهد به‌صورت موازی به روابط مختلف نگاه کند.

سرهای مختلف ممکن است در الگوهایی مانند این تخصص پیدا کنند:

- توکن‌های نزدیک
- وابستگی‌های بلندبرد
- نحو (Syntax)
- نام‌های تکرارشونده
- فهرست‌ها
- تورفتگی کد (Code Indentation)
- نشانه‌گذاری
- ساختار سند
- رفتار شبه‌بازیابی (Retrieval-like Copying)

---

## ۴.۱۶ ابعاد سرهای attention

معمولاً داریم:

$$
d_{\text{model}} = n_{\text{heads}} \times d_{\text{head}}
$$

برای مثال:

```text
d_model = 4096
n_heads = 32
d_head = 128
```
projectionهای attention معمولاً از این نگاشت استفاده می‌کنند:

$$
d_{\text{model}} \rightarrow n_{\text{heads}} \times d_{\text{head}}
$$

انتخاب بُعد سر attention بر این موارد اثر می‌گذارد:

- کیفیت attention
- چیدمان حافظه (Memory Layout)
- کارایی کرنل‌ها (Kernel Efficiency)
- اندازه‌ی KV-cache
- رفتار RoPE
- بهره‌وری سخت‌افزاری

بسیاری از LLMهای مدرن از ابعاد سر مانند 64 یا 128 استفاده می‌کنند.

---

## ۴.۱۷ شبکه‌ی پیش‌خور (Feed-Forward Network)

هر بلوک ترنسفورمر شامل یک شبکه‌ی پیش‌خورِ position-wise نیز هست.

FFN به‌صورت مستقل روی هر موقعیت توکن اعمال می‌شود.

یک FFN کلاسیک به‌صورت زیر است:

$$
\text{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2
$$

که در آن:

-  $W_1$ از  $d_{\text{model}}$ به  $d_{\text{ff}}$ projection می‌کند.
-  $W_2$ از  $d_{\text{ff}}$ دوباره به  $d_{\text{model}}$ projection می‌کند.
-  $\sigma$ یک تابع غیرخطی مانند ReLU، GELU یا SiLU است.

معمولاً:

$$
d_{\text{ff}} \approx 4 \times d_{\text{model}}
$$

اگرچه FFNهای gated مدرن اغلب از نسبت‌های متفاوتی استفاده می‌کنند.

FFN سهم زیادی از تعداد پارامترها و هزینه‌ی محاسباتی مدل را بر عهده دارد.

---

## ۴.۱۸ توابع فعال‌سازی

ترنسفورمرهای اولیه از ReLU استفاده می‌کردند. مدل‌های بعدی اغلب از GELU بهره گرفتند.

LLMهای مدرن معمولاً از فعال‌سازهای gated مانند **SwiGLU** استفاده می‌کنند.

### ReLU

$$
\text{ReLU}(x) = \max(0, x)
$$

### GELU

GELU نسبت به ReLU نرم‌تر است و در مدل‌هایی مانند BERT و GPT-2 استفاده شد.

### SiLU

$$
\text{SiLU}(x) = x \cdot \sigma(x)
$$

که در آن $\sigma$ تابع sigmoid است.

### SwiGLU

SwiGLU یک فعال‌ساز gated است:

$$
\text{SwiGLU}(x) = \text{SiLU}(xW_g) \odot (xW_u)
$$

و سپس یک down projection روی آن اعمال می‌شود.

یک فرم ساده‌شده‌ی آن:

$$
\text{FFN}(x) = (\text{SiLU}(xW_g) \odot xW_u)W_d
$$

که در آن:

- $W_g$ projection مربوط به gate است.
- $W_u$ projection بالارونده (Up Projection) است.
- $W_d$ projection پایین‌رونده (Down Projection) است.
- $\odot$ ضرب درایه‌به‌درایه (Elementwise Multiplication) است.

SwiGLU به‌طور گسترده در LLMهای مدرن فقط-رمزگشا استفاده می‌شود.

---

## ۴.۱۹ اتصال‌های باقیمانده

اتصال باقیمانده، ورودی یک زیرلایه را به خروجی آن اضافه می‌کند:

$$
y = x + f(x)
$$

در یک بلوک ترنسفورمر:

```text
x -> attention -> add back to x
x -> FFN       -> add back to x
```
اتصال‌های باقیمانده به این دلایل مهم‌اند:

- بهبود جریان گرادیان
- پایدارسازی شبکه‌های عمیق
- حفظ اطلاعات در طول لایه‌ها
- ساده‌تر شدن بهینه‌سازی

بدون اتصال‌های باقیمانده، آموزش ترنسفورمرهای بسیار عمیق بسیار دشوارتر می‌شد.

---

## ۴.۲۰ لایه‌های نرمال‌سازی

نرمال‌سازی باعث پایداری آموزش می‌شود.

ترنسفورمر اولیه از LayerNorm پس از زیرلایه‌ها استفاده می‌کرد. در LLMهای مدرن اغلب از Pre-LN یا RMSNorm استفاده می‌شود.

### LayerNorm

LayerNorm روی ابعاد نهان برای هر توکن نرمال‌سازی انجام می‌دهد:

$$
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

که در آن:

- $\mu$ میانگین روی ابعاد نهان است.
- $\sigma^2$ واریانس است.
- $\gamma$ و $\beta$ پارامترهای آموختنی‌اند.

### RMSNorm

RMSNorm مؤلفه‌ی کم‌کردن میانگین را حذف می‌کند و بر اساس ریشه‌ی میانگین مربعات نرمال‌سازی می‌کند:

$$
\text{RMSNorm}(x) = \gamma \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}}
$$

RMSNorm ساده‌تر و اغلب سریع‌تر از LayerNorm است.

بسیاری از LLMهای مدرن از RMSNorm استفاده می‌کنند.

---

## ۴.۲۱ Pre-LN در برابر Post-LN

دو جای‌گذاری رایج برای نرمال‌سازی وجود دارد.

### Post-LN

در ترنسفورمر اولیه استفاده می‌شد:

$$
H' = \text{Norm}(H + \text{Sublayer}(H))
$$

### Pre-LN

در LLMهای مدرن رایج‌تر است:

$$
H' = H + \text{Sublayer}(\text{Norm}(H))
$$

Pre-LN پایداری آموزش را برای مدل‌های عمیق‌تر بهبود می‌دهد.

### تفاوت عملی

Post-LN می‌تواند کار کند، اما ترنسفورمرهای عمیقِ Post-LN اغلب بدون initialization دقیق یا schedule مناسبِ learning rate سخت‌تر آموزش می‌بینند.

برای LLMهای بزرگ فقط-رمزگشا، معمولاً Pre-LN ترجیح داده می‌شود.

---

## ۴.۲۲ ترنسفورمر فقط-رمزگشا

بیشتر LLMهای مولد مدرن از ترنسفورمر فقط-رمزگشا (Decoder-only Transformer) استفاده می‌کنند.

نمونه‌ها شامل مدل‌های سبک GPT و بسیاری از خانواده‌های LLM متن‌باز هستند.

یک مدل فقط-رمزگشا از این اجزا استفاده می‌کند:

- embeddingهای توکن
- اطلاعات موقعیتی
- بلوک‌های self-attention علّیِ پشته‌شده
- شبکه‌های پیش‌خور
- نرمال‌سازی نهایی
- projection خروجی به logits واژگان

ویژگی کلیدی آن، attention علّی است:

```text
هر توکن فقط می‌تواند به خودش و توکن‌های قبلی توجه کند.
```
این ویژگی به‌طور مستقیم با پیش‌بینی توکن بعدی سازگار است.

## ۴.۲۳ ترنسفورمر فقط-رمزگذار

ترنسفورمرهای فقط-رمزگذار (Encoder-only Transformers)، مانند مدل‌های سبک BERT، از توجه دوسویه (Bidirectional Attention) استفاده می‌کنند.

هر توکن می‌تواند به همه‌ی توکن‌ها توجه کند:
```text
key1   key2   key3   key4
query1     ✓      ✓      ✓      ✓
query2     ✓      ✓      ✓      ✓
query3     ✓      ✓      ✓      ✓
query4     ✓      ✓      ✓      ✓
```
مدل‌های فقط-رمزگذار برای این کاربردها مفید هستند:

- Classification
- Embeddings
- Retrieval
- Reranking
- Named entity recognition
- Token classification
- Semantic similarity

این مدل‌ها معمولاً با اهدافی مانند مدل‌سازی زبانِ ماسک‌شده (Masked Language Modeling) آموزش می‌بینند.

آن‌ها معماری استاندارد برای تولید خودبازگشتی متن (Autoregressive Text Generation) نیستند.

---

## ۴.۲۴ ترنسفورمر رمزگذار-رمزگشا

ترنسفورمرهای رمزگذار-رمزگشا (Encoder-decoder Transformers) در ترنسفورمر اولیه استفاده شدند و در مدل‌های sequence-to-sequence مانند T5 رایج هستند.

این معماری شامل این اجزاست:

- یک رمزگذار (Encoder) که دنباله‌ی ورودی را به‌صورت دوسویه می‌خواند.
- یک رمزگشا (Decoder) که خروجی را به‌صورت خودبازگشتی تولید می‌کند.
- Cross-attention از توکن‌های رمزگشا به خروجی‌های رمزگذار.

این معماری برای این وظایف مفید است:

- Translation
- Summarization
- Text-to-text tasks
- Conditional generation

رمزگشا دارای دو سازوکار attention است:

- Causal self-attention روی توکن‌های تولیدشده
- Cross-attention روی حالت‌های رمزگذار

بسیاری از LLMهای چتِ همه‌منظوره‌ی مدرن، به‌جای این معماری از معماری فقط-رمزگشا استفاده می‌کنند، زیرا مدل‌های فقط-رمزگشا به‌سادگی مقیاس می‌گیرند و برای وظایف مولد گسترده عملکرد خوبی دارند.

---

## ۴.۲۵ چرا مدل‌های فقط-رمزگشا بر LLMها غالب‌اند؟

ترنسفورمرهای فقط-رمزگشا (Decoder-only Transformers) برای LLMها محبوب‌اند، زیرا:

- Simple
- Scalable
- Compatible with next-token prediction
- Easy to train on raw text
- Easy to adapt to chat formatting
- Efficient for autoregressive generation
- Flexible across tasks through prompting

قالب آموزش آن‌ها ساده است:

```text
Document tokens:
x1 x2 x3 ... xT

Training:
predict x2 from x1
predict x3 from x1 x2
predict x4 from x1 x2 x3
...
```
همین هدف آموزشی برای این نوع داده‌ها نیز کار می‌کند:

- Web documents
- Books
- Code
- Conversations
- Tool traces
- Mathematical text
- Instruction data

این سادگی یکی از دلایل اصلی است که ترنسفورمرهای فقط-رمزگشا به انتخاب پیش‌فرض برای مدل‌های مولد بزرگ تبدیل شدند.

---

## ۴.۲۶ Projection خروجی و Logits

پس از آخرین لایه‌ی ترنسفورمر، هر موقعیت یک hidden state دارد:

$$
h_t \in \mathbb{R}^{d_{\text{model}}}
$$

برای پیش‌بینی توکن بعدی، مدل این بردار را به اندازه‌ی واژگان (Vocabulary Size) projection می‌کند:

$$
z_t = h_t W_{\text{out}} + b
$$

که در آن:

$$
W_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times |V|}
$$

خروجی $z_t$. بردار logits نامیده می‌شود:

$$
z_t \in \mathbb{R}^{|V|}
$$

توزیع احتمال به‌صورت زیر است:

$$
P(x_{t+1} \mid x_{\le t}) = \text{softmax}(z_t)
$$

loss آموزشی، cross-entropy نسبت به توکن صحیحِ بعدی است.

---

## ۴.۲۷ Weight Tying

برخی مدل‌های زبانی، ماتریس embedding ورودی و ماتریس projection خروجی را با هم share می‌کنند.

به این کار weight tying گفته می‌شود.

embedding ورودی:

$$
E \in \mathbb{R}^{|V| \times d_{\text{model}}}
$$

projection خروجی می‌تواند از این رابطه استفاده کند:

$$
W_{\text{out}} = E^\top
$$

weight tying تعداد پارامترها را کاهش می‌دهد و می‌تواند کارایی را بهبود دهد.

با این حال، همه‌ی معماری‌ها وزن‌ها را tie نمی‌کنند. این تصمیم به طراحی مدل، setup آموزشی و جزئیات پیاده‌سازی بستگی دارد.

---

## ۴.۲۸ پیچیدگی محاسباتی Attention

ماتریس امتیاز attention دارای شکل زیر است:

$$
T \times T
$$

برای هر لایه و هر head، attention نیاز دارد هر توکن را با هر توکنِ مجاز دیگر مقایسه کند.

پیچیدگی تقریبی attention برابر است با:

$$
O(T^2 d)
$$

که در آن:

- $T$ طول دنباله (Sequence Length) است.
- $d$ بسته به زمینه، اندازه‌ی hidden state یا بُعد head است.

این وابستگی درجه‌دو به طول دنباله، یکی از محدودیت‌های اصلی ترنسفورمرها است.

اگر طول دنباله دو برابر شود، محاسبه‌ی امتیازهای attention و مصرف حافظه تقریباً چهار برابر رشد می‌کند.

به همین دلیل، آموزش و استنتاج long-context به مهندسی دقیق نیاز دارد.

---

## ۴.۲۹ پیچیدگی محاسباتی FFN

شبکه‌ی پیش‌خور (Feed-Forward Network) به‌صورت مستقل روی هر موقعیت توکن اعمال می‌شود.

پیچیدگی تقریبی آن برابر است با:

$$
O(T d_{\text{model}} d_{\text{ff}})
$$

از آن‌جا که $d_{\text{ff}}$ معمولاً چند برابرِ $d_{\text{model}}$ است، FFN می‌تواند بخش غالبِ محاسبه باشد، به‌ویژه در طول‌دنباله‌های متوسط.

در بسیاری از LLMها، بخش بزرگی از پارامترها در لایه‌های FFN قرار دارد.

به همین دلیل، طراحی FFN هم برای کیفیت مدل و هم برای بهره‌وری آموزشی اهمیت دارد.

---

## ۴.۳۰ تعداد پارامترهای یک بلوک ترنسفورمر

یک بلوک ساده‌شده‌ی ترنسفورمر فقط-رمزگشا شامل این اجزاست:

- Query projection
- Key projection
- Value projection
- Output attention projection
- FFN up projection
- FFN down projection
- Normalization parameters

اگر biasها و normalization را نادیده بگیریم، attention چندسریِ استاندارد تقریباً این تعداد پارامتر دارد:

$$
4 d_{\text{model}}^2
$$

این تعداد مربوط به این ماتریس‌ها است:

- $W_Q$
- $W_K$
- $W_V$
- $W_O$

یک FFN استاندارد تقریباً این تعداد پارامتر دارد:

$$
2 d_{\text{model}} d_{\text{ff}}
$$

اگر:

$$
d_{\text{ff}} = 4d_{\text{model}}
$$

در آن صورت پارامترهای FFN برابر می‌شود با:

$$
8d_{\text{model}}^2
$$

بنابراین، یک بلوک کلاسیک ترنسفورمر تقریباً این تعداد پارامتر دارد:

$$ 
12d_{\text{model}}^2
$$

که embeddingها و جمله‌های کوچک‌تر را شامل نمی‌شود.

برای FFNهای gated مانند SwiGLU، شمارش پارامتر متفاوت است، زیرا معمولاً سه ماتریس FFN وجود دارد: projection مربوط به gate، projection بالارونده (Up) و projection پایین‌رونده (Down).

---

## ۴.۳۱ KV Cache

در استنتاج خودبازگشتی (Autoregressive Inference)، مدل در هر گام یک توکن تولید می‌کند.

برای هر توکن تولیدشده، هر لایه بردارهای key و value را محاسبه می‌کند.

به‌جای آن‌که در هر گام، keyها و valueهای تمام توکن‌های قبلی دوباره محاسبه شوند، سیستم‌های استنتاج آن‌ها را در یک **KV cache** ذخیره می‌کنند.

در گام تولید $t$، مدل این موارد را ذخیره می‌کند:

```text
Keys:   K_1, K_2, ..., K_t
Values: V_1, V_2, ..., V_t
```
برای توکن بعدی، مدل فقط query، key و value جدید را محاسبه می‌کند و سپس به keyها و valueهای cache‌شده‌ی قبلی attention می‌دهد.

این کار decoding را بسیار سریع‌تر می‌کند.

---

## ۴.۳۲ هزینه‌ی حافظه‌ی KV Cache

حافظه‌ی KV cache با این عوامل رشد می‌کند:

- Batch size
- Sequence length
- Number of layers
- Number of KV heads
- Head dimension
- Precision

یک فرمول ساده‌شده به‌صورت زیر است:

$$
\text{KV cache elements} = 2 \times B \times T \times L \times n_{\text{kv heads}} \times d_{\text{head}}
$$

که در آن:

- $2$ مربوط به keyها و valueها است.
- $B$ اندازه‌ی batch است.
- $T$ طول دنباله است.
- $L$ تعداد لایه‌ها است.
- $n_{\text{kv heads}}$ تعداد headهای key-value است.
- $d_{\text{head}}$ بُعد head است.

حافظه بر حسب بایت:

$$
\text{bytes} = \text{elements} \times \text{bytes per element}
$$

در سروینگ long-context، حافظه‌ی KV-cache اغلب یک گلوگاه اصلی است، زیرا به‌صورت خطی با طول دنباله، batch size، تعداد لایه‌ها، تعداد KV headها و بُعد head رشد می‌کند.

---
## ۴.۳۳ Multi-Query Attention (توجه چندپرس‌وجویی)

Multi-Query Attention (MQA) (توجه چندپرس‌وجویی) از تعداد زیادی Query Head (سر پرس‌وجو) استفاده می‌کند، اما یک Key Head (سر کلید) و یک Value Head (سر مقدار) را میان آن Query Headها به‌اشتراک می‌گذارد.

Standard Multi-Head Attention (توجه چندسر استاندارد):

```text
Q heads: many
K heads: many
V heads: many
```
MQA:

```text
Q heads: many
K heads: one
V heads: one
```
این روش اندازه KV Cache (حافظه نهان کلید-مقدار) را هنگام Inference (استنتاج) به‌طور چشمگیری کاهش می‌دهد.


### Advantages (مزایا)

- حافظه کمتر برای KV Cache (حافظه نهان کلید-مقدار)
- Decoding (رمزگشایی) سریع‌تر
- Serving Efficiency (بهره‌وری ارائه مدل) بهتر


### Disadvantages (معایب)

- اگر بیش‌ازحد استفاده شود، ممکن است کیفیت مدل را کاهش دهد.
- در مقایسه با بازنمایی‌های کامل Multi-Head KV (کلید-مقدار چندسر)، Expressiveness (بیان‌پذیری) کمتری دارد.

MQA (توجه چندپرس‌وجویی) عمدتاً یک بهینه‌سازی برای Inference Efficiency (بهره‌وری استنتاج) است.

---


## ۴.۳۴ Grouped-Query Attention (توجه گروهی-پرس‌وجویی)

Grouped-Query Attention (GQA) (توجه گروهی-پرس‌وجویی)، یا GQA، یک راه‌حل میانه میان Standard Multi-Head Attention (توجه چندسر استاندارد) و MQA (توجه چندپرس‌وجویی) است.

این روش از تعداد زیادی Query Head (سر پرس‌وجو)، اما تعداد کمتری Key-Value Head (سر کلید-مقدار) استفاده می‌کند.

Example (مثال):

```text
Query heads: 32
KV heads: 8

Each group of query heads shares one key-value head.

GQA reduces KV cache memory while preserving more capacity than MQA.
```
بسیاری از LLMهای مدرن (مدل‌های زبانی بزرگ) از GQA استفاده می‌کنند، زیرا این روش تعادل مناسبی میان Quality (کیفیت) و Inference Efficiency (بهره‌وری استنتاج) فراهم می‌کند.

---


## ۴.۳۵ Sliding Window Attention (توجه پنجره لغزان)

برخی مدل‌ها برای کاهش هزینه Attention (توجه) در دنباله‌های طولانی، از Sliding Window Attention (توجه پنجره لغزان) استفاده می‌کنند.

به‌جای اینکه هر Token (توکن) بتواند به تمام Tokenهای قبلی توجه کند، هر Token فقط به یک پنجره اخیر توجه می‌کند.

Example (مثال):

‍‍‍‍```text
Window size = 4

Token 10 attends to tokens:
7, 8, 9, 10
```

این روش هزینه محاسباتی Full Quadratic Attention (توجه درجه‌دوم کامل) را در سراسر دنباله کاهش می‌دهد.


### Advantages (مزایا)

- مصرف حافظه کمتر
- Long-Context Scalability (مقیاس‌پذیری بهتر برای متن طولانی)
- آموزش و Inference (استنتاج) سریع‌تر برای دنباله‌های طولانی


### Disadvantages (معایب)

- دسترسی مستقیم به Tokenهای دور را محدود می‌کند.
- به تنظیمات معماری یا آموزشی نیاز دارد.
- ممکن است به وظایفی که به Long-Range Dependencies (وابستگی‌های دوربرد) نیاز دارند، آسیب بزند.

برخی معماری‌ها Local Attention (توجه محلی) را با Global Attention (توجه سراسری) گاه‌به‌گاه یا سازوکارهای دیگر ترکیب می‌کنند.

---


## ۴.۳۶ Attention Implementation Details (جزئیات پیاده‌سازی توجه)

Naive Attention (توجه ساده) کل Attention Matrix (ماتریس توجه) را در حافظه ایجاد می‌کند:


$$
T \times T
$$


این موضوع می‌تواند هزینه حافظه زیادی داشته باشد.

پیاده‌سازی‌های مدرن از Optimized Attention Kernels (هسته‌های بهینه توجه)، مانند FlashAttention، استفاده می‌کنند.

این Kernelها با محاسبه Attention (توجه) در Blockها (بلوک‌ها) و جلوگیری از ایجاد کامل Attention Matrix (ماتریس توجه)، Memory Overhead (سربار حافظه) را کاهش می‌دهند.

Benefits (مزایا) شامل موارد زیر است:

- مصرف حافظه کمتر
- آموزش سریع‌تر
- استفاده بهتر از GPU
- پشتیبانی از دنباله‌های طولانی‌تر

Kernelهای مشابه FlashAttention همان عملیات Standard Scaled Dot-Product Attention (توجه ضرب داخلی مقیاس‌یافته استاندارد) را محاسبه می‌کنند، اما با استفاده از IO-Aware Tiling (قطعه‌بندی آگاه از ورودی/خروجی)، ترافیک پرهزینه حافظه را کاهش داده و از ایجاد کامل Attention Matrix (ماتریس توجه) جلوگیری می‌کنند.

---


## ۴.۳۷ Dropout (دراپ‌اوت)

Dropout (دراپ‌اوت) در زمان Training (آموزش)، Activationها (فعال‌سازی‌ها) را به‌صورت تصادفی صفر می‌کند.

Original Transformer (ترنسفورمر اولیه) در چندین بخش از Dropout استفاده می‌کرد:

- Attention Weights (وزن‌های توجه)
- Residual Connections (اتصال‌های باقی‌مانده)
- Feed-Forward Activations (فعال‌سازی‌های شبکه پیش‌خور)
- Embeddings (تعبیه‌ها)

بسیاری از LLMهای مدرن در مقیاس بزرگ در زمان Pretraining (پیش‌آموزش)، به‌ویژه هنگامی که روی Datasetهای (مجموعه‌داده‌های) بسیار بزرگ آموزش می‌بینند، از مقدار کمی Dropout یا اصلاً از Dropout استفاده نمی‌کنند.

بااین‌حال، Dropout همچنان می‌تواند برای موارد زیر مفید باشد:

- مدل‌های کوچک‌تر
- مجموعه‌داده‌های کوچک‌تر
- Fine-Tuning (ریزتنظیم)
- جلوگیری از Overfitting (بیش‌برازش)

Dropout در زمان Inference (استنتاج) غیرفعال است.

---


## ۴.۳۸ Initialization (مقداردهی اولیه)

Weight Initialization (مقداردهی اولیه وزن‌ها) بر Training Stability (پایداری آموزش) تأثیر می‌گذارد.

Poor Initialization (مقداردهی اولیه نامناسب) می‌تواند باعث موارد زیر شود:

- Exploding Activations (فعال‌سازی‌های انفجاری)
- Vanishing Gradients (گرادیان‌های ناپدیدشونده)
- Slow Convergence (همگرایی کند)
- ناپایداری در شبکه‌های عمیق

Large Transformers (ترنسفورمرهای بزرگ) اغلب از Initialization Schemeهای (طرح‌های مقداردهی اولیه) با دقت انتخاب‌شده استفاده می‌کنند که گاهی شامل Depth Scaling (مقیاس‌دهی وابسته به عمق) نیز می‌شوند.

طرح دقیق به عوامل زیر وابسته است:

- Normalization Placement (محل قرارگیری نرمال‌سازی)
- Activation Function (تابع فعال‌سازی)
- Model Depth (عمق مدل)
- Residual Path Design (طراحی مسیر باقی‌مانده)
- Optimizer (بهینه‌ساز)
- Precision Format (قالب دقت عددی)

Initialization موضوعی است که به‌سادگی نادیده گرفته می‌شود، اما در مقیاس بالا اهمیت زیادی دارد.

---


## ۴.۳۹ Residual Stream (جریان باقی‌مانده)

تفسیرهای مدرن از Transformers (ترنسفورمرها) اغلب Hidden State (حالت پنهان) را به‌عنوان یک **Residual Stream (جریان باقی‌مانده)** توصیف می‌کنند.

هر Layer (لایه) از این Stream (جریان) می‌خواند، یک Update (به‌روزرسانی) محاسبه می‌کند و نتیجه را دوباره در آن می‌نویسد.

Simplified (نمایش ساده‌شده):

```text
Residual stream
   |
   +-- attention reads and writes update
   |
   +-- FFN reads and writes update
   |
   +-- next layer continues
```
این دیدگاه مفید است، زیرا اطلاعات می‌تواند در سراسر شبکه و از میان Layerهای (لایه‌های) متعدد جریان پیدا کند؛ درحالی‌که هر Block (بلوک)، Transformationهایی (تبدیل‌هایی) را روی Shared Representation (بازنمایی مشترک) اعمال می‌کند.

---


## ۴.۴۰ Depth and Width (عمق و عرض)

Transformer Capacity (ظرفیت ترنسفورمر) به‌شدت به عوامل زیر وابسته است:

- Number of Layers (تعداد لایه‌ها)
- Hidden Dimension (بُعد پنهان)
- Number of Attention Heads (تعداد سرهای توجه)
- FFN Dimension (بُعد شبکه پیش‌خور)
- Vocabulary Size (اندازه واژگان)
- Context Length (طول متن زمینه)

Depth (عمق) به معنای تعداد بیشتر Transformer Block (بلوک ترنسفورمر) است.

Width (عرض) به معنای بزرگ‌تر بودن Hidden Dimension (بُعد پنهان) است.

افزایش Depth و Width هر دو Parameter Count (تعداد پارامترها) و Compute (محاسبات) را افزایش می‌دهند، اما بر Learning (یادگیری) تأثیر متفاوتی دارند.


### Deeper Models (مدل‌های عمیق‌تر)

Advantages (مزایا):

- Sequential Transformations (تبدیل‌های ترتیبی) بیشتر
- امکان دستیابی به Abstraction (انتزاع) بهتر
- Compositional Processing (پردازش ترکیبی) بیشتر

Disadvantages (معایب):

- Optimization (بهینه‌سازی) دشوارتر
- Latency (تأخیر) بیشتر
- Activation Memory (حافظه فعال‌سازی) بیشتر
- نیاز بیشتر به Stable Normalization and Initialization (نرمال‌سازی و مقداردهی اولیه پایدار)


### Wider Models (مدل‌های عریض‌تر)

Advantages (مزایا):

- Capacity (ظرفیت) بیشتر در هر Layer (لایه)
- Parallelism (موازی‌سازی) بهتر در برخی Regimeهای (رژیم‌های) سخت‌افزاری
- Representation Space (فضای بازنمایی) بزرگ‌تر

Disadvantages (معایب):

- Matrix Multiplication Cost (هزینه ضرب ماتریسی) بیشتر
- Attention و FFN Projectionهای (پروجکشن‌های) بزرگ‌تر
- فشار بیشتر بر Memory Bandwidth (پهنای باند حافظه)

Scaling Laws (قوانین مقیاس‌پذیری) به هدایت این انتخاب‌ها کمک می‌کنند، اما Architecture (معماری) و Hardware Constraints (محدودیت‌های سخت‌افزاری) نیز اهمیت دارند.

---


## ۴.۴۱ Context Length (طول متن زمینه)

Context Length (طول متن زمینه) حداکثر تعداد Tokenهایی (توکن‌هایی) است که مدل می‌تواند در یک زمان پردازش کند.

برای یک Decoder-Only LLM (مدل زبانی بزرگ فقط-رمزگشا)، Context Length بر موارد زیر تأثیر می‌گذارد:

- Maximum Prompt Size (حداکثر اندازه پرامپت)
- Training Sequence Length (طول دنباله آموزشی)
- Attention Memory (حافظه موردنیاز توجه)
- KV-Cache Size (اندازه حافظه نهان کلید-مقدار)
- Long-Document Ability (توانایی پردازش اسناد طولانی)
- RAG Document Packing (بسته‌بندی اسناد در RAG)
- Chat History Length (طول تاریخچه گفتگو)

Context Length برابر با 8192 Token به معنای 8192 Word (کلمه) نیست. همان‌طور که در Chapter 3 (فصل ۳) توضیح داده شد، Tokenization (توکن‌سازی) تعیین می‌کند چه مقدار متن در این فضا جای می‌گیرد.

Context طولانی‌تر مفید است، اما هزینه زیادی دارد.

---


## ۴.۴۲ Long-Context Challenges (چالش‌های متن طولانی)

Long-Context Models (مدل‌های دارای متن زمینه طولانی) با چندین چالش مواجه هستند:

- Quadratic Attention Cost (هزینه درجه‌دوم توجه)
- Large KV-Cache Memory (حافظه زیاد موردنیاز KV Cache)
- Positional Extrapolation (برون‌یابی موقعیتی)
- Data Scarcity for Long Sequences (کمبود داده برای دنباله‌های طولانی)
- Difficulty Using Distant Information (دشواری استفاده از اطلاعات دور)
- Evaluation Complexity (پیچیدگی ارزیابی)
- Retrieval-Like Failure Modes (الگوهای شکست مشابه بازیابی)
- Increased Serving Cost (افزایش هزینه ارائه مدل)

صرفاً افزایش Configured Context Window (پنجره متن زمینه پیکربندی‌شده) تضمین نمی‌کند که مدل بتواند به‌طور قابل‌اعتماد از اطلاعات موجود در تمام Positionها (موقعیت‌ها) استفاده کند.

Long-Context Capability (قابلیت پردازش متن طولانی) به عوامل زیر وابسته است:

- Architecture (معماری)
- Positional Encoding (کدگذاری موقعیتی)
- Training Data (داده‌های آموزشی)
- sequence-length curriculum (برنامه آموزشی طول دنباله)
- Attention Implementation (پیاده‌سازی توجه)
- Evaluation Design (طراحی ارزیابی)

---


## ۴.۴۳ Positional Extrapolation (برون‌یابی موقعیتی)

مدلی که روی دنباله‌های 4096-Token آموزش دیده است، ممکن است در 32768 Token عملکرد خوبی نداشته باشد.

دلایل این موضوع شامل موارد زیر است:

- Positional Embeddings (تعبیه‌های موقعیتی) ممکن است از Positionهای (موقعیت‌های) مشاهده‌نشده پشتیبانی نکنند.
- Attention Patterns (الگوهای توجه) ممکن است تعمیم پیدا نکنند.
- Training Data (داده‌های آموزشی) ممکن است شامل Long-Range Dependencies (وابستگی‌های دوربرد) نباشد.
- RoPE Frequencies (فرکانس‌های RoPE) ممکن است فراتر از Trained Range (محدوده آموزش‌دیده) عملکرد نامناسبی داشته باشند.

Long-Context Extension Methods (روش‌های گسترش متن زمینه طولانی) اغلب RoPE Scaling (مقیاس‌دهی RoPE) را تغییر می‌دهند یا مدل را روی دنباله‌های طولانی‌تر Fine-Tune (ریزتنظیم) می‌کنند.

بااین‌حال، Context Extension (گسترش متن زمینه) باید با دقت ارزیابی شود. ممکن است یک مدل بتواند یک Long Prompt (پرامپت طولانی) را بپذیرد، اما نتواند به‌طور قابل‌اعتماد از Far-Away Information (اطلاعات دوردست) استفاده کند.

---


## ۴.۴۴ Model Configuration Example (مثال پیکربندی مدل)

یک Transformer Configuration (پیکربندی ترنسفورمر) می‌تواند به شکل زیر باشد:

```json
{
  "vocab_size": 50000,
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 11008,
  "hidden_act": "silu",
  "max_position_embeddings": 8192,
  "rope_theta": 10000,
  "rms_norm_eps": 1e-6,
  "tie_word_embeddings": false
}
```
Important Fields (فیلدهای مهم) شامل موارد زیر هستند:

- `vocab_size`: تعداد Tokenهای (توکن‌های) Tokenizer (توکنایزر).
- `hidden_size`: Model Width (عرض مدل).
- `num_hidden_layers`: Model Depth (عمق مدل).
- `num_attention_heads`: تعداد Query Headها (سرهای پرس‌وجو).
- `num_key_value_heads`: تعداد o

[قبلی: توکن‌سازی](./03-tokenization.md) |
[فهرست مطالب](./index.md) |
[بعدی: پیش‌آموزش](./05-pretraining.md)
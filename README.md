# University AI Assistant

مساعد ذكي بيجاوب على أسئلة الطلاب عن اللوائح الجامعية ومتطلبات التخرج، مبني بالكامل
باستخدام تقنيات المنهج: RAG + LangChain + موديل محلي، مع Fine-tuning و Quantization
و Output Parsing و Deployment.

## طريقة الربط بكل ملف من الكورس

| الجزء | الملف عندك في الكورس | الملف المقابل هنا |
|---|---|---|
| قراءة PDF وتقسيمه | — | `src/document_loader.py` (بـ `pdftotext`، مش `PyPDFLoader` — شرح تحت) |
| RAG (embeddings + FAISS) | `update-rag.ipynb` | `src/vector_store.py`, `src/retriever.py` |
| موديل محلي | `Mistral.ipynb` | `src/llm.py` |
| Quantization | `Quantization.ipynb` | `src/llm.py` (`USE_4BIT_QUANTIZATION`) |
| Output Parser | `Update_OutputParser__1___2_.ipynb` | `src/output_parser.py` |
| Fine-tuning (LoRA) | `FineTuning_NewKnowledge.ipynb`, `fine-tune-summary__1_.ipynb` | `src/finetune.py` |
| Deployment (FastAPI + ngrok) | `NGROK_Kaggle.ipynb`, `NGROK_Send_Prompt_Local.ipynb` | `deploy_server.py`, `client_example.py` |

## ⚠️ ملاحظة مهمة عن قراءة الـ PDF

اللوائح غالبًا هتكون بالعربي، و`PyPDFLoader` بتاع LangChain (بيستخدم مكتبة `pypdf`)
**بيطلع العربي مبعثر وغير مرتب** (حروف وكلمات معكوسة). جربتها بنفسي على مستند
اللائحة وطلعت نتيجة غير قابلة للاستخدام. البديل اللي شغال صح هو أداة `pdftotext`
(من حزمة `poppler-utils`)، فـ `document_loader.py` بيستخدمها بدل `PyPDFLoader`.
لو محتاج تتأكد بنفسك، جرب:
```bash
python -m src.document_loader
```
وشوف الـ sample chunk في الآخر — المفروض يطلع عربي سليم ومرتب.

## هيكل المشروع

```
university-ai-assistant/
├── src/
│   ├── config.py          # كل الإعدادات (موديلز، quantization، chunk size)
│   ├── document_loader.py # قراءة PDFs (pdftotext) وتقسيمها chunks
│   ├── vector_store.py    # embeddings + FAISS
│   ├── retriever.py       # البحث + confidence score
│   ├── llm.py             # تحميل الموديل المحلي (مع/بدون quantization)
│   ├── chain.py           # الـ RAG pipeline الكامل مع structured output
│   ├── output_parser.py   # الـ StructuredOutputParser (JSON schema)
│   ├── finetune.py        # LoRA fine-tuning بسيط
│   └── evaluation.py      # تقييم بسيط لجودة الإجابات
├── app.py                 # واجهة Streamlit (محلي)
├── deploy_server.py       # FastAPI + ngrok (نشر على Kaggle GPU)
├── client_example.py      # مثال لمناداة السيرفر المنشور
├── data/
│   ├── sadat_academy_cs_regulations_sample.pdf  # لائحة تمثيلية جاهزة
│   └── qa_finetune_sample.json                   # أمثلة Q&A للـ fine-tuning
├── requirements.txt
└── .env.example
```

## التشغيل — مهم تعرفه الأول

الموديل (Mistral-7B) والـ embeddings بيحتاجوا **GPU** عشان يشتغلوا في وقت معقول.
لو معاك لابتوب من غير GPU قوي، الطريقة الأفضل زي بالظبط اللي عملتها في الكورس:

1. شغل الكود على **Kaggle Notebook** (فيه GPU مجاني T4)
2. استخدم `deploy_server.py` عشان تنشر الموديل بـ ngrok
3. من جهازك الشخصي، استخدم `client_example.py` أو `app.py` (بعد ما تعدله يستخدم الـ API بدل الموديل المحلي مباشرة) عشان تكلم السيرفر

## التشغيل خطوة بخطوة (على Kaggle أو أي بيئة فيها GPU)

1. تنصيب المكتبات:
```bash
pip install -r requirements.txt
```

2. اختبار قراءة الـ PDF (سريع، مش محتاج GPU):
```bash
python -m src.document_loader
```

3. بناء الـ FAISS index (محتاج تحميل موديل الـ embeddings أول مرة):
```bash
python -m src.vector_store
```

4. اختبار الموديل المحلي والـ quantization (بياخد وقت، بيحمل الموديل):
```bash
python -m src.llm
```

5. سؤال تجريبي من الـ RAG chain الكامل:
```bash
python -m src.chain
```

6. تشغيل التقييم:
```bash
python -m src.evaluation
```

7. (اختياري) الـ fine-tuning:
```bash
python -m src.finetune
```

8. تشغيل الواجهة محليًا (لو معاك GPU على نفس الجهاز):
```bash
streamlit run app.py
```

9. أو نشر السيرفر (على Kaggle) والاتصال بيه من جهازك:
```bash
# على Kaggle:
python deploy_server.py
# على جهازك، بعد ما تحط الـ ngrok URL في client_example.py:
python client_example.py
```

## المميزات

- **RAG بموديل محلي**: مافيش اعتماد على API خارجي، كل حاجة شغالة بالوزن المحلي للموديل
- **Quantization**: تقدر تقارن حجم/سرعة الموديل بـ 4-bit مقابل fp16 (`src/llm.py::benchmark()`)
- **Structured Output**: كل إجابة بترجع JSON فيه `answer` + `source_page` + `confidence`
- **Fine-tuning اختياري**: LoRA بسيط لتحسين أسلوب الرد (مش لحفظ المعلومات، ده شغل الـ RAG)
- **Deployment**: FastAPI + ngrok، بالظبط زي اللي اتعلمته
- **Evaluation Harness**: مجموعة أسئلة تقييمية بتقيس نسبة الإجابات الصحيحة ومتوسط الثقة

## ملاحظات للتطوير

- عدّل `TEST_SET` في `src/evaluation.py` بأسئلة حقيقية من اللائحة عشان تطلع أرقام تقييم فعلية
- عدّل `data/qa_finetune_sample.json` وزوّد الأمثلة (50-100 مثال) قبل تشغيل `finetune.py` فعليًا
- بعد الـ fine-tuning، غيّر `LLM_MODEL_NAME` في `config.py` لمسار الموديل المدمج
  (`outputs/university-assistant-lora/merged`) عشان الـ chain يستخدم النسخة المحسّنة
- `CHUNK_SIZE`، `TOP_K_RESULTS`، و `USE_4BIT_QUANTIZATION` في `config.py` قيم تستاهل
  تجربها وتقيس تأثيرها على الجودة/السرعة — وده مادة كويسة لقسم التقييم في التقرير

import re
from languageUtils import LanguageUtils
from llm_client import LLMClient
from typing import List, Dict, Tuple

class AnswerGenerator:
    """
    Generate grounded answers using retrieved context.

    Key principles:
    1. ONLY use information from provided sources
    2. ALWAYS cite sources for factual claims
    3. If information not found, say so clearly
    4. Can synthesize/compare if info exists in sources
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def get_system_prompt(self, lang: str) -> str:
        """Get system prompt based on language."""
        if lang == "ar":
            return """أنت مساعد تعليمي ذكي. مهمتك هي الإجابة على أسئلة الطلاب باستخدام المصادر المقدمة فقط.

## القواعد الأساسية:

### 1. استخدم المصادر فقط
- أجب فقط باستخدام المعلومات الموجودة في المصادر المقدمة
- لا تستخدم معرفتك العامة أبداً
- إذا لم تجد المعلومة: قل "هذه المعلومات غير متوفرة في المواد المقدمة"

### 2. الاستشهاد إلزامي
- ضع [رقم] بعد كل معلومة مباشرة
- مثال: "RGB هو نموذج لوني [1]"
- لا تذكر معلومة بدون استشهاد

### 3. أسئلة المقارنة
- إذا طُلب مقارنة موضوعين:
  - ابحث عن كل موضوع في المصادر
  - إذا وجدت كليهما: قارن بينهما مع الاستشهاد لكل معلومة
  - إذا وجدت أحدهما فقط: اشرح ما وجدته واذكر أن الآخر غير موجود
  - إذا لم تجد أياً منهما: قل ذلك بوضوح

### 4. التنسيق
- اكتب بالعربية الفصحى
- المصطلحات التقنية: بالعربية ثم (English)
- فقرات واضحة ومترابطة
- 100-200 كلمة تقريباً"""

        else:
            return """You are an intelligent tutoring assistant. Your task is to answer student questions using ONLY the provided sources.

## Core Rules:

### 1. Use Sources Only
- Answer ONLY using information from the provided sources
- NEVER use your general knowledge
- If information not found: Say "This information is not available in the provided materials"

### 2. Citations are Mandatory
- Add [number] after every factual statement
- Example: "RGB is a color model [1]"
- Never state a fact without citation

### 3. Comparison Questions
- If asked to compare two topics:
  - Search for each topic in the sources
  - If BOTH found: Compare them with citations for each fact
  - If only ONE found: Explain what you found and state the other is not in the materials
  - If NEITHER found: State this clearly

### 4. Synthesis is Allowed
- You CAN combine information from different sources
- You CAN explain relationships between concepts if the individual concepts are in sources
- You CAN use reasoning to answer based on source content
- But EVERY fact must be cited

### 5. Format
- Clear, flowing paragraphs
- Academic but accessible
- About 100-200 words"""

    def build_context(self, retrieved_docs: List[Dict]) -> Tuple[str, List[Dict]]:
        """Build context string and source list from retrieved documents."""
        context_parts = []
        sources = []

        for i, doc in enumerate(retrieved_docs, 1):
            # Use parent text for richer context
            text = doc["metadata"].get("parent_text", doc["document"])
            context_parts.append(f"[Source {i}]:\n{text}")

            sources.append(
                {
                    "source": doc["metadata"].get("source", "Unknown"),
                    "page": doc["metadata"].get("page", "?"),
                    "lang": doc["metadata"].get("lang", "en"),
                    "text": text,
                }
            )

        return "\n\n".join(context_parts), sources

    def generate(
        self, query: str, retrieved_docs: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """Generate answer from retrieved documents."""
        lang = LanguageUtils.detect_language(query)

        if not retrieved_docs:
            no_info = (
                "لم أجد معلومات ذات صلة في المواد المرفوعة. يرجى التأكد من رفع الملفات المناسبة."
                if lang == "ar"
                else "I couldn't find relevant information in the uploaded materials. Please make sure to upload the appropriate files."
            )
            return no_info, []

        # Build context
        context, sources = self.build_context(retrieved_docs)

        # Get prompts
        system_prompt = self.get_system_prompt(lang)

        if lang == "ar":
            user_prompt = f"""المصادر المتاحة:
{context}

سؤال الطالب: {query}

تذكر:
- استخدم المصادر فقط
- استشهد بكل معلومة [رقم]
- إذا المعلومة غير موجودة، قل ذلك"""
        else:
            user_prompt = f"""Available Sources:
{context}

Student Question: {query}

Remember:
- Use ONLY the sources above
- Cite every fact [number]
- If information is not found, say so clearly"""

        # Generate answer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer = self.llm_client.chat(messages, temperature=0.3, max_tokens=1024)

        # Add sources section
        if lang == "ar":
            sources_header = "\n\nالمصادر:"
            source_lines = [
                f"[{i}] {s['source']} - صفحة {s['page']}"
                for i, s in enumerate(sources, 1)
            ]
        else:
            sources_header = "\n\nSources:"
            source_lines = [
                f"[{i}] {s['source']} - page {s['page']}"
                for i, s in enumerate(sources, 1)
            ]

        answer = answer.strip() + sources_header + "\n" + "\n".join(source_lines)

        return answer, sources

    def generate_followups(self, query: str, answer: str, lang: str) -> List[str]:
        """Generate follow-up questions."""
        if lang == "ar":
            prompt = """بناءً على السؤال والجواب، اقترح 3 أسئلة متابعة قصيرة بالعربية.
اكتب الأسئلة فقط، سؤال في كل سطر، بدون ترقيم.

السؤال: {query}
الجواب: {answer}"""
        else:
            prompt = """Based on this Q&A, suggest 3 short follow-up questions in English.
Write only the questions, one per line, without numbering.

Question: {query}
Answer: {answer}"""

        try:
            result = self.llm_client.chat(
                [
                    {
                        "role": "user",
                        "content": prompt.format(query=query, answer=answer[:500]),
                    }
                ],
                temperature=0.7,
                max_tokens=200,
            )

            lines = [l.strip() for l in result.split("\n") if l.strip()]
            questions = []
            for line in lines[:3]:
                # Clean numbering
                line = re.sub(r"^[\d\.\-\*\)]+\s*", "", line)
                if line and len(line) > 10:
                    questions.append(line)
            return questions
        except:
            return []
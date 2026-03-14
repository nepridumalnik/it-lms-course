# %% [markdown]
# # Задание 9. Введение в NLP. Парсинг и суммаризация статьи с Habr
#
# ## Задания
# 1. Парсинг веб-страницы
# 2. Структурированная суммаризация
# 3. Абстрактная суммаризация
# 4. Сравнение результатов

# %% [markdown]
# ## Импорты и конфигурация

# %%
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


@dataclass(frozen=True)
class Config:
    base_dir: Path = Path(__file__).resolve().parent
    root_dir: Path = base_dir.parent
    env_path: Path = root_dir / ".env"
    article_url: str = "https://habr.com/ru/articles/923096/"
    request_timeout_seconds: int = 30
    max_tokens: int = 50000
    article_path: Path = base_dir / "article.txt"
    structured_summary_path: Path = base_dir / "structured_summary.txt"
    structured_quality_path: Path = base_dir / "structured_quality.txt"
    abstract_summary_path: Path = base_dir / "abstract_summary.txt"
    comparison_path: Path = base_dir / "summaries_comparison.txt"


@dataclass(frozen=True)
class APIConfig:
    api_key: str
    base_url: str
    model: str


CFG = Config()


# %% [markdown]
# ## Вспомогательные функции

# %%
def _fallback_load_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def load_api_config(env_path: Path) -> APIConfig:
    if load_dotenv is not None:
        load_dotenv(env_path, override=False)
    else:
        _fallback_load_env(env_path)

    api_key = os.getenv("API_KEY")
    base_url = os.getenv("API_URL")
    model = os.getenv("API_MODEL")

    missing = [
        name
        for name, value in (
            ("API_KEY", api_key),
            ("API_URL", base_url),
            ("API_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "В .env отсутствуют обязательные переменные: " + ", ".join(missing)
        )

    return APIConfig(
        api_key=api_key or "",
        base_url=base_url or "",
        model=model or "",
    )


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def fetch_article_html(url: str, timeout_seconds: int) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def _clean_lines(lines: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        normalized = re.sub(r"\s+", " ", line).strip()
        if len(normalized) >= 2:
            cleaned.append(normalized)
    return cleaned


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    article_root = (
        soup.select_one("article.tm-article-presenter__content")
        or soup.select_one(".tm-article-body")
        or soup.select_one("article")
        or soup.body
    )
    if article_root is None:
        raise RuntimeError("Не удалось найти тело статьи на странице.")

    for node in article_root.select(
        "script, style, noscript, iframe, svg, form, button, input, nav, aside, footer"
    ):
        node.decompose()

    text = article_root.get_text(separator="\n")
    lines = _clean_lines(text.splitlines())
    if not lines:
        raise RuntimeError("После очистки статья оказалась пустой.")

    return "\n".join(lines)


def _extract_message_text(message_content: object) -> str:
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()

    return ""


def ask_llm(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        max_tokens=CFG.max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = _extract_message_text(response.choices[0].message.content)
    if not answer:
        raise RuntimeError("Модель вернула пустой ответ.")
    return answer.strip()


STRUCTURED_SCHEMA = """\
Ответ строго в следующей структуре:

ТЕМА:
...

КРАТКОЕ РЕЗЮМЕ (2-3 предложения):
...

КЛЮЧЕВЫЕ ИДЕИ:
- ...
- ...
- ...

ИНСТРУМЕНТЫ/ПОДХОДЫ:
- ...

ВЫВОДЫ АВТОРА:
- ...

ПРАКТИЧЕСКАЯ ПОЛЬЗА:
...

ОГРАНИЧЕНИЯ/РИСКИ:
...
"""


def build_structured_summary(client: OpenAI, model: str, article_text: str) -> str:
    system_prompt = (
        "Ты аналитик технических статей. "
        "Пиши только по содержанию входного текста, без выдумок."
    )
    user_prompt = (
        "Сделай структурированную суммаризацию статьи.\n"
        "Строго соблюдай шаблон ниже и заполни каждый раздел.\n\n"
        f"{STRUCTURED_SCHEMA}\n\n"
        "Текст статьи:\n"
        f"{article_text}"
    )
    return ask_llm(client, model, system_prompt, user_prompt)


def build_abstract_summary(client: OpenAI, model: str, article_text: str) -> str:
    system_prompt = (
        "Ты делаешь абстрактную суммаризацию. "
        "Нужен один связный пересказ простыми словами."
    )
    user_prompt = (
        "Сделай свободный пересказ статьи по условиям:\n"
        "- один цельный текст;\n"
        "- без списков;\n"
        "- без заголовков.\n\n"
        "Текст статьи:\n"
        f"{article_text}"
    )
    return ask_llm(client, model, system_prompt, user_prompt)


REQUIRED_BLOCKS = [
    "ТЕМА:",
    "КРАТКОЕ РЕЗЮМЕ",
    "КЛЮЧЕВЫЕ ИДЕИ:",
    "ИНСТРУМЕНТЫ/ПОДХОДЫ:",
    "ВЫВОДЫ АВТОРА:",
    "ПРАКТИЧЕСКАЯ ПОЛЬЗА:",
    "ОГРАНИЧЕНИЯ/РИСКИ:",
]


def evaluate_structured_summary(article_text: str, structured_summary: str) -> str:
    missing = [block for block in REQUIRED_BLOCKS if block not in structured_summary]
    article_len = len(article_text)
    summary_len = len(structured_summary)
    compression = summary_len / max(article_len, 1)

    if missing:
        structure_comment = (
            "Структура соблюдена частично: не найдены разделы -> "
            + ", ".join(missing)
            + "."
        )
    else:
        structure_comment = (
            "Структура соблюдена полностью: все обязательные разделы на месте."
        )

    if compression < 0.08:
        detail_comment = (
            "Суммаризация очень краткая; часть деталей могла быть потеряна."
        )
    elif compression > 0.5:
        detail_comment = (
            "Суммаризация слишком объёмная; стоит сильнее сжать материал."
        )
    else:
        detail_comment = (
            "Объём суммаризации выглядит сбалансированным для обзора статьи."
        )

    return (
        f"{structure_comment}\n"
        f"{detail_comment}\n"
        f"Доля объёма summary/article: {compression:.2f}."
    )


def compare_summaries(structured_summary: str, abstract_summary: str) -> str:
    structured_has_lists = "- " in structured_summary
    abstract_has_lists = "- " in abstract_summary

    abstract_style = (
        "Абстрактный вариант выполнен в виде цельного текста без списков."
        if not abstract_has_lists
        else "Абстрактный вариант содержит списки, что не полностью соответствует требованию."
    )
    structured_style = (
        "Структурированный вариант удобен для быстрого поиска фактов по разделам."
        if structured_has_lists
        else "Структурированный вариант менее формализован, чем ожидалось."
    )

    return (
        f"{structured_style}\n"
        f"{abstract_style}\n"
        "На практике структурированный формат лучше для отчёта и проверки критериев, "
        "а абстрактный пересказ удобнее для быстрого чтения и общего понимания."
    )


# %% [markdown]
# # Инициализация API-клиента
#
# Подтягиваем значения из `.env`:
# - `API_KEY`
# - `API_URL`
# - `API_MODEL`

# %%
api_cfg = load_api_config(CFG.env_path)
client = OpenAI(api_key=api_cfg.api_key, base_url=api_cfg.base_url)

print("API-клиент инициализирован.")
print("Используемая модель:", api_cfg.model)


# %% [markdown]
# # Задание 1. Парсинг веб-страницы
#
# Скачиваем статью, очищаем от HTML/шума и сохраняем в `article.txt`.

# %%
html = fetch_article_html(CFG.article_url, CFG.request_timeout_seconds)
article_text = extract_article_text(html)
save_text(CFG.article_path, article_text)

print("Файл статьи сохранён:", CFG.article_path)
print("Количество символов в очищенном тексте:", len(article_text))


# %%
print("Фрагмент article.txt:\n")
print(article_text[:2000])


# %% [markdown]
# # Задание 2. Структурированная суммаризация
#
# Формируем ответ строго по заданной схеме и сохраняем в `structured_summary.txt`.

# %%
structured_summary = build_structured_summary(client, api_cfg.model, article_text)
save_text(CFG.structured_summary_path, structured_summary)

print("Файл структурированной суммаризации сохранён:", CFG.structured_summary_path)
print()
print(structured_summary)


# %% [markdown]
# ## Вывод о качестве структурированной суммаризации

# %%
quality_comment = evaluate_structured_summary(article_text, structured_summary)
save_text(CFG.structured_quality_path, quality_comment)

print("Файл с выводом о качестве сохранён:", CFG.structured_quality_path)
print()
print(quality_comment)


# %% [markdown]
# # Задание 3. Абстрактная суммаризация
#
# Делаем свободный пересказ без списков и заголовков, сохраняем в `abstract_summary.txt`.

# %%
abstract_summary = build_abstract_summary(client, api_cfg.model, article_text)
save_text(CFG.abstract_summary_path, abstract_summary)

print("Файл абстрактной суммаризации сохранён:", CFG.abstract_summary_path)
print()
print(abstract_summary)


# %% [markdown]
# # Задание 4. Сравнение результатов
#
# Сравниваем структурированную и абстрактную суммаризации.

# %%
comparison = compare_summaries(structured_summary, abstract_summary)
save_text(CFG.comparison_path, comparison)

print("Файл сравнения сохранён:", CFG.comparison_path)
print()
print(comparison)


# %% [markdown]
# # Итог
#
# Все требуемые артефакты сформированы:
# - `article.txt`
# - `structured_summary.txt`
# - `structured_quality.txt`
# - `abstract_summary.txt`
# - `summaries_comparison.txt`

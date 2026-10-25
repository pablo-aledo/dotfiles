#!/usr/bin/env python3
"""Lee un prompt por stdin, lo envía a chatgpt.com (sin login) con Selenium
y escribe la respuesta por stdout. Equivalente a `gpt-ai`.

Dependencias: pip install undetected-chromedriver selenium
Variables opcionales:
  GPT_HEADLESS=1         -> fuerza headless (se activa solo si no hay DISPLAY)
  GPT_PROFILE=<dir>      -> perfil persistente de Chrome (por defecto ~/.cache/gpt_web_profile)
  GPT_TIMEOUT=300        -> segundos máximos esperando respuesta
  GPT_CHROME=<ruta>      -> binario de Chrome
  GPT_CHROME_VERSION=155 -> versión mayor de Chrome
  GPT_DEBUG=1            -> vuelca diagnóstico aunque no haya error
"""
import os
import sys
import time
import warnings

warnings.filterwarnings(
    "ignore", message="urllib3 .* or chardet .* doesn't match a supported version"
)

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

URLS = [
    "https://chatgpt.com/?temporary-chat=true",  # chat temporal: no queda en historial
    "https://chatgpt.com/",
]
TIMEOUT = int(os.environ.get("GPT_TIMEOUT", "300"))
PROFILE = os.environ.get("GPT_PROFILE", os.path.expanduser("~/.cache/gpt_web_profile"))
POLL = 0.5

# Varios selectores por si la web cambia; se prueban en orden.
# Los primeros son el diseño "de escritorio" clásico y los siguientes el
# diseño nuevo (mobile shell) que sirve ChatGPT desde 2025.
INPUT_SELS = [
    "#prompt-textarea",
    "textarea#mobile-composer-prompt",
    "textarea[name='prompt']",
    "textarea[aria-label='Chat with ChatGPT']",
    "div.ProseMirror[contenteditable='true']",
    "[contenteditable='true']",
    "textarea",
    "[role='textbox']",
]
SEND_SELS = [
    "button[data-testid='send-button']",
    "button[data-composer-submit]",
    "button[aria-label='Send message']",
    "button[aria-label*='Send']",
    "button[aria-label*='Enviar']",
]
STOP_SELS = [
    "button[data-testid='stop-button']",
    "button[aria-label*='Stop']",
    "button[aria-label*='Detener']",
]
ASSISTANT_SELS = [
    "div[data-message-author-role='assistant']",
    "[data-message-role='assistant']",
]
USER_SELS = [
    "div[data-message-author-role='user']",
    "[data-message-role='user']",
]


def log(msg):
    print(msg, file=sys.stderr)


def make_driver():
    opts = uc.ChromeOptions()
    opts.add_argument(f"--user-data-dir={PROFILE}")
    opts.add_argument("--window-size=1400,1000")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    headless = os.environ.get("GPT_HEADLESS") == "1" or not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if headless:
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        # Evita que el User-Agent delate "HeadlessChrome"
        opts.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36"
        )
    kwargs = {}
    binary = os.environ.get("GPT_CHROME")
    if binary:
        kwargs["browser_executable_path"] = binary
    ver = os.environ.get("GPT_CHROME_VERSION")
    if ver:
        kwargs["version_main"] = int(ver)
    # use_subprocess es imprescindible con Chrome reciente: sin él Chrome
    # arranca pero el driver no consigue conectarse ("chrome not reachable").
    return uc.Chrome(options=opts, use_subprocess=True, **kwargs)


def first_displayed(driver, selectors):
    for sel in selectors:
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            try:
                if el.is_displayed():
                    return el
            except Exception:
                pass
    return None


def wait_for(driver, selectors, timeout):
    end = time.time() + timeout
    while time.time() < end:
        el = first_displayed(driver, selectors)
        if el is not None:
            return el
        time.sleep(POLL)
    return None


def dismiss_popups(driver):
    """Cierra avisos de cookies / 'Stay logged out' si aparecen."""
    xpaths = [
        "//a[contains(., 'Stay logged out') or contains(., 'Permanecer sin sesión')]",
        "//button[contains(., 'Stay logged out') or contains(., 'Permanecer sin sesión')]",
        "//button[contains(., 'Reject non-essential') or contains(., 'Rechazar')]",
        "//button[contains(., 'Stay logged out')]",
    ]
    for xp in xpaths:
        for el in driver.find_elements(By.XPATH, xp):
            try:
                el.click()
                time.sleep(0.5)
            except Exception:
                pass


def dump_diagnostics(driver, tag="error"):
    try:
        log(f"URL: {driver.current_url}  Título: {driver.title!r}")
        body = driver.find_element(By.TAG_NAME, "body").text.strip()
        log("--- Texto visible (primeros 600 caracteres) ---")
        log(body[:600] or "(vacío)")
        log("--- Campos de entrada encontrados ---")
        for sel in INPUT_SELS:
            n = len(driver.find_elements(By.CSS_SELECTOR, sel))
            log(f"  {sel}: {n}")
        log("--- Botones (aria-label / texto) ---")
        for b in driver.find_elements(By.TAG_NAME, "button")[:25]:
            label = b.get_attribute("aria-label") or b.text.strip()
            tid = b.get_attribute("data-testid")
            if label or tid:
                log(f"  {label!r} testid={tid!r}")
        png, html = f"/tmp/gpt_{tag}.png", f"/tmp/gpt_{tag}.html"
        driver.save_screenshot(png)
        with open(html, "w") as f:
            f.write(driver.page_source)
        log(f"Captura en {png} y HTML en {html}")
    except Exception as e:
        log(f"(no se pudo volcar diagnóstico: {e})")


def open_chat(driver):
    """Abre la web y espera al cuadro de texto; prueba URLs alternativas."""
    for url in URLS:
        driver.get(url)
        time.sleep(3)
        dismiss_popups(driver)
        box = wait_for(driver, INPUT_SELS, 40)
        if box is not None:
            return box
        log(f"Sin cuadro de texto en {url}")
        dump_diagnostics(driver, "noinput")
    raise RuntimeError("No se encontró el cuadro de texto del chat")


def send_prompt(driver, box, text):
    box.click()
    time.sleep(0.3)
    try:
        box.send_keys(text)
    except Exception:
        # insertText conserva los saltos de línea sin que Enter envíe el mensaje
        driver.execute_script(
            "arguments[0].focus(); document.execCommand('insertText', false, arguments[1]);",
            box,
            text,
        )
    time.sleep(0.5)
    # El botón de enviar está deshabilitado mientras el cuadro está vacío;
    # esperamos a que se habilite antes de pulsarlo.
    send = None
    end = time.time() + 15
    while time.time() < end:
        send = first_displayed(driver, SEND_SELS)
        if send is not None:
            disabled = (send.get_attribute("aria-disabled") or "").lower() == "true"
            if not disabled:
                break
        time.sleep(0.3)
    if send is not None:
        try:
            send.click()
        except Exception:
            driver.execute_script("arguments[0].click();", send)
    else:
        log("Botón de enviar no encontrado; pulso Enter")
        box.send_keys(Keys.ENTER)


def find_assistant_messages(driver):
    els = []
    for sel in ASSISTANT_SELS:
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            if el not in els:
                els.append(el)
    return els


def assistant_text(el):
    """Texto limpio del mensaje (sin 'ChatGPT said:' ni botones de acción)."""
    try:
        if el.get_attribute("data-conversation-recovery") is not None:
            return ""
    except Exception:
        pass
    for sel in (
        "[data-assistant-markdown-root]",
        "[data-assistant-markdown]",
        ".markdown",
    ):
        try:
            roots = el.find_elements(By.CSS_SELECTOR, sel)
        except Exception:
            roots = []
        for root in reversed(roots):
            try:
                t = root.text.strip()
            except Exception:
                t = ""
            if t:
                return t
    try:
        return el.text.strip()
    except Exception:
        return ""


def get_last_answer(driver):
    answer = ""
    for el in find_assistant_messages(driver):
        try:
            if not el.is_displayed():
                continue
        except Exception:
            continue
        t = assistant_text(el)
        if t:
            answer = t
    return answer


def count_answers(driver):
    n = 0
    for el in find_assistant_messages(driver):
        try:
            if el.is_displayed() and assistant_text(el):
                n += 1
        except Exception:
            pass
    return n


def count_messages(driver, selectors):
    n = 0
    for sel in selectors:
        n = max(n, len(driver.find_elements(By.CSS_SELECTOR, sel)))
    return n


def is_generating(driver):
    for sel in STOP_SELS:
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            try:
                if el.is_displayed():
                    return True
            except Exception:
                pass
    return False


def wait_for_answer(driver, baseline_answer="", baseline_count=0):
    start = time.time()
    # Fase 1: esperar a que empiece una respuesta nueva (streaming o texto nuevo).
    while time.time() - start < TIMEOUT:
        cur = get_last_answer(driver)
        if is_generating(driver):
            break
        if cur and (cur != baseline_answer or count_answers(driver) > baseline_count):
            break
        time.sleep(POLL)
    else:
        raise TimeoutError("No apareció respuesta")

    # Fase 2: esperar a que termine de generar (sin botón stop y texto estable).
    last, stable = "", 0
    while time.time() - start < TIMEOUT:
        cur = get_last_answer(driver)
        generating = is_generating(driver)
        is_new = cur != baseline_answer or count_answers(driver) > baseline_count
        if cur and is_new and cur == last and not generating:
            stable += 1
            if stable >= 4:  # ~2 s estable
                return cur
        else:
            stable = 0
        last = cur
        time.sleep(POLL)
    raise TimeoutError("Timeout esperando el final de la respuesta")


def main():
    prompt = sys.stdin.read().strip()
    if not prompt:
        log("Prompt vacío")
        sys.exit(1)
    driver = make_driver()
    try:
        box = open_chat(driver)
        baseline_answer = get_last_answer(driver)
        baseline_count = count_answers(driver)
        send_prompt(driver, box, prompt)
        answer = wait_for_answer(driver, baseline_answer, baseline_count)
        if os.environ.get("GPT_DEBUG") == "1":
            dump_diagnostics(driver, "debug")
        print(answer)
    except Exception as e:
        log(f"Error: {type(e).__name__}: {e}")
        dump_diagnostics(driver, "error")
        sys.exit(2)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()

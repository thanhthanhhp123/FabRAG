import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUTPUT_DIR = "datasheets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 50 documents cho electronic-component / datasheet RAG corpus
# ============================================================

URLS = {
    # # ========================================================
    # # STMicroelectronics — STM32
    # # ========================================================
    # "01_STM32F103C4_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32f103c4.pdf",

    # "02_STM32F103CB_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32f103cb.pdf",

    # "03_STM32F103RC_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32f103rc.pdf",

    # "04_STM32F10xxx_reference_manual.pdf":
    #     "https://www.st.com/resource/en/reference_manual/cd00171190-stm32f101xx-stm32f102xx-stm32f103xx-stm32f105xx-and-stm32f107xx-advanced-arm-based-32-bit-mcus-stmicroelectronics.pdf",

    # "05_STM32F401RE_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32f401re.pdf",

    # "06_STM32F407VG_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32f407vg.pdf",

    # "07_STM32F411CE_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32f411ce.pdf",

    # "08_STM32F446RE_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32f446re.pdf",

    # "09_STM32G431RB_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32g431rb.pdf",

    # "10_STM32L476RG_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32l476rg.pdf",

    # "11_STM32H743ZI_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32h743zi.pdf",

    # "12_STM32F030C6_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32f030c6.pdf",

    # "13_STM32F072RB_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32f072rb.pdf",

    # "14_STM32G071RB_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32g071rb.pdf",

    # "15_STM32L432KC_datasheet.pdf":
    #     "https://www.st.com/resource/en/datasheet/stm32l432kc.pdf",


    # # ========================================================
    # # Texas Instruments — analog / sensor / ADC / logic
    # # ========================================================
    # "16_LM358_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/lm358.pdf",

    # "17_LM324_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/lm324.pdf",

    # "18_LM393_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/lm393.pdf",

    # "19_TL072_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/tl072.pdf",

    # "20_TL082_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/tl082.pdf",

    # "21_NE555_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/ne555.pdf",

    # "22_UA741_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/ua741.pdf",

    # "23_LM317_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/lm317.pdf",

    # "24_INA219_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/ina219.pdf",

    # "25_INA226_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/ina226.pdf",

    # "26_ADS1115_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/ads1115.pdf",

    # "27_ADS1015_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/ads1015.pdf",

    # "28_TPS5430_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/tps5430.pdf",

    # "29_SN74HC595_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/sn74hc595.pdf",

    # "30_MAX232_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/max232.pdf",

    # "31_DRV8833_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/drv8833.pdf",

    # "32_L293_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/l293.pdf",

    # "33_SN74HC165_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/sn74hc165.pdf",

    # "34_CD4051B_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/cd4051b.pdf",

    # "35_TCA9548A_datasheet.pdf":
    #     "https://www.ti.com/lit/ds/symlink/tca9548a.pdf",


    # ========================================================
    # Analog Devices — instrumentation / sensors / converters
    # ========================================================
    "36_INA333_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/ina333.pdf",

"37_OPA333_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/opa333.pdf",

"38_TLV9002_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/tlv9002.pdf",

"39_TMP117_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/tmp117.pdf",

"40_TMP102_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/tmp102.pdf",

"41_TMP112_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/tmp112.pdf",

"42_INA180_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/ina180.pdf",

"43_INA181_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/ina181.pdf",

"44_INA240_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/ina240.pdf",

"45_OPA197_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/opa197.pdf",

"46_OPA2197_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/opa2197.pdf",

"47_TLV9062_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/tlv9062.pdf",

"48_MCP3008_REPLACEMENT_ADC081S101_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/adc081s101.pdf",

"49_DAC8562_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/dac8562.pdf",

"50_LM75B_datasheet.pdf":
    "https://www.ti.com/lit/ds/symlink/lm75b.pdf",
}


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*;q=0.8",
}


def create_session():
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=10,
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def is_pdf(content):
    """
    PDF thật luôn bắt đầu bằng magic bytes:
        %PDF-
    """
    return content[:5] == b"%PDF-"


def download_all():
    session = create_session()

    total = len(URLS)
    success = 0
    failed = []

    print("=" * 70)
    print(f"Datasheet downloader")
    print(f"Target : {total} files")
    print(f"Output : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)

    for index, (filename, url) in enumerate(URLS.items(), start=1):

        path = os.path.join(OUTPUT_DIR, filename)

        print(f"\n[{index:02d}/{total}] {filename}")

        # ----------------------------------------------------
        # Skip file hợp lệ đã tồn tại
        # ----------------------------------------------------
        if os.path.exists(path):

            try:
                with open(path, "rb") as f:
                    header = f.read(5)

                if header == b"%PDF-":
                    size_mb = os.path.getsize(path) / 1024 / 1024
                    print(f"  [skip] already exists ({size_mb:.2f} MB)")
                    success += 1
                    continue

                else:
                    print("  [warning] existing file is not PDF -> redownload")
                    os.remove(path)

            except Exception:
                pass

        try:
            print(f"  [GET] {url}")

            response = session.get(
                url,
                headers=HEADERS,
                timeout=(15, 90),
                allow_redirects=True,
            )

            response.raise_for_status()

            content = response.content

            # ------------------------------------------------
            # Quan trọng:
            # Đừng chỉ tin Content-Type.
            # Một số server trả application/octet-stream.
            # Kiểm tra magic bytes mới chắc chắn.
            # ------------------------------------------------
            if not is_pdf(content):
                content_type = response.headers.get(
                    "Content-Type",
                    "unknown"
                )

                raise ValueError(
                    f"response is NOT a PDF "
                    f"(Content-Type={content_type}, "
                    f"bytes={len(content)})"
                )

            # file quá bé thường có vấn đề
            if len(content) < 10_000:
                raise ValueError(
                    f"PDF suspiciously small: {len(content)} bytes"
                )

            # ghi .part trước để tránh file corrupt
            temp_path = path + ".part"

            with open(temp_path, "wb") as f:
                f.write(content)

            os.replace(temp_path, path)

            size_mb = len(content) / 1024 / 1024

            print(
                f"  [OK] {size_mb:.2f} MB"
            )

            success += 1

            # tránh rate-limit
            time.sleep(1)

        except Exception as e:

            print(
                f"  [FAILED] {type(e).__name__}: {e}"
            )

            failed.append(
                {
                    "filename": filename,
                    "url": url,
                    "error": str(e),
                }
            )

            # xóa file tạm nếu có
            temp_path = path + ".part"

            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

            time.sleep(2)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print(f"Total   : {total}")
    print(f"Success : {success}")
    print(f"Failed  : {len(failed)}")

    if failed:

        failed_txt = os.path.join(
            OUTPUT_DIR,
            "_failed_downloads.txt"
        )

        with open(
            failed_txt,
            "w",
            encoding="utf-8"
        ) as f:

            for item in failed:
                f.write(
                    f"{item['filename']}\n"
                    f"{item['url']}\n"
                    f"ERROR: {item['error']}\n\n"
                )

        print(
            f"\nFailed list saved to:\n"
            f"  {failed_txt}"
        )

        print("\nFailed files:")

        for item in failed:
            print(
                f"  - {item['filename']}"
            )

    else:
        print("\nAll 50 documents downloaded successfully.")

    print(
        f"\nDatasheets folder:\n"
        f"  {os.path.abspath(OUTPUT_DIR)}"
    )


if __name__ == "__main__":
    download_all()
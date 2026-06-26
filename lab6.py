#!/usr/bin/env python3
"""
LAB6: Algorytm YOLO
Pełna implementacja
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ============================================================
# TODO 1: Import klasy YOLO z biblioteki ultralytics
# ============================================================
from ultralytics import YOLO


# ============================================================
# TODO 2: Wczytanie modelu bazowego
# ============================================================

def wczytaj_model(sciezka_modelu: str):
    """
    Wczytuje model YOLO z pliku.
    Jeśli plik nie istnieje lokalnie, ultralytics pobierze go automatycznie
    (np. yolov8n.pt zostanie ściągnięty z serwerów Ultralytics).
    """
    try:
        model = YOLO(sciezka_modelu)
        print(f"[OK] Model wczytany: {sciezka_modelu}")
        return model
    except Exception as e:
        print(f"[BŁĄD] Nie udało się wczytać modelu '{sciezka_modelu}': {e}")
        sys.exit(1)


# ============================================================
# TODO 3: Wczytanie obrazu
# ============================================================

def wczytaj_obraz(sciezka_obrazu: str) -> np.ndarray:
    """
    Wczytuje obraz z pliku za pomocą OpenCV.
    Sprawdza poprawność wczytania i kończy program w przypadku błędu.
    """
    obraz = cv2.imread(sciezka_obrazu)
    if obraz is None:
        print(f"[BŁĄD] Nie można wczytać obrazu: '{sciezka_obrazu}'")
        sys.exit(1)
    print(f"[OK] Obraz wczytany: {sciezka_obrazu}  rozmiar={obraz.shape[1]}x{obraz.shape[0]}")
    return obraz


# ============================================================
# TODO 4: Wczytanie strumienia wideo
# ============================================================

def wczytaj_strumien(zrodlo: str):
    """
    Otwiera strumień wideo: kamera ('camera') lub plik wideo.
    Sprawdza poprawność otwarcia i kończy program w przypadku błędu.
    """
    if zrodlo == "camera":
        # Otwórz domyślną kamerę (indeks 0)
        cap = cv2.VideoCapture(0)
        nazwa = "kamera"
    else:
        # Otwórz plik wideo podany jako ścieżka
        cap = cv2.VideoCapture(zrodlo)
        nazwa = f"plik '{zrodlo}'"

    if not cap.isOpened():
        print(f"[BŁĄD] Nie można otworzyć: {nazwa}")
        sys.exit(1)

    print(f"[OK] Strumień wideo otwarty: {nazwa}")
    return cap


# ============================================================
# TODO 5: Uruchomienie detekcji
# ============================================================

def wykonaj_detekcje(model, obraz: np.ndarray):
    """
    Uruchamia model YOLO na pojedynczym obrazie lub klatce wideo.
    Parametr verbose=False wyłącza szczegółowe logi modelu.
    Zwraca surowe wyniki modelu.
    """
    wyniki = model(obraz, verbose=False)
    return wyniki


# ============================================================
# TODO 6: Odczyt danych z wyników YOLO
# ============================================================

def odczytaj_wyniki(wyniki) -> List[Dict[str, Any]]:
    """
    Odczytuje dane detekcji z wyników modelu YOLO.

    Zwracana lista detekcji zawiera słowniki:
    [
        {
            "box":        [x1, y1, x2, y2],
            "class_id":   int,
            "confidence": float
        },
        ...
    ]
    """
    detekcje = []
    result = wyniki[0]  # bierzemy pierwszy (i zazwyczaj jedyny) wynik

    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # współrzędne [x1, y1, x2, y2]
        conf = float(box.conf[0].cpu().numpy())  # wartość pewności (confidence score)
        cls = int(box.cls[0].cpu().numpy())  # identyfikator klasy

        detekcje.append({
            "box": xyxy.tolist(),
            "class_id": cls,
            "confidence": conf,
        })

    return detekcje


# ============================================================
# TODO 7: Filtrowanie detekcji po progu confidence
# ============================================================

def filtruj_detekcje(
        detekcje: List[Dict[str, Any]],
        prog_confidence: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Usuwa detekcje z wartością confidence poniżej progu.
    Model YOLO stosuje NMS wewnętrznie, więc tutaj wystarczy filtr progowy.
    """
    return [d for d in detekcje if d["confidence"] >= prog_confidence]


# ============================================================
# TODO 8: Pobranie nazw klas z modelu
# ============================================================

def pobierz_nazwy_klas(model) -> Optional[Dict[int, str]]:
    """
    Pobiera słownik {class_id: "nazwa_klasy"} wbudowany w model Ultralytics.
    Zwraca None, jeśli model nie zawiera nazw klas.
    """
    try:
        return model.names  # słownik wbudowany w model
    except AttributeError:
        return None


# ============================================================
# TODO 9: Rysowanie detekcji na obrazie
# ============================================================

def rysuj_detekcje(
        obraz: np.ndarray,
        detekcje: List[Dict[str, Any]],
        nazwy_klas: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """
    Rysuje bounding boxy i etykiety (nazwa klasy + confidence) na kopii obrazu.
    Każda klasa otrzymuje unikalny kolor generowany deterministycznie.
    """
    wynik = obraz.copy()

    for det in detekcje:
        x1, y1, x2, y2 = map(int, det["box"])
        cls = det["class_id"]
        conf = det["confidence"]

        # Ustal nazwę klasy lub użyj identyfikatora liczbowego
        if nazwy_klas and cls in nazwy_klas:
            label = nazwy_klas[cls]
        else:
            label = str(cls)

        # Unikalny kolor dla każdej klasy w formacie BGR
        color = tuple(int(c) for c in np.random.default_rng(cls).integers(80, 255, size=3))

        # Narysuj prostokąt wokół wykrytego obiektu
        cv2.rectangle(wynik, (x1, y1), (x2, y2), color, 2)

        # Przygotuj tekst etykiety z wartością confidence
        text = f"{label}: {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

        # Narysuj tło pod tekstem dla lepszej czytelności
        cv2.rectangle(wynik, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)

        # Narysuj tekst etykiety
        cv2.putText(
            wynik, text,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    return wynik


# ============================================================
# TODO 10: Napisy diagnostyczne
# ============================================================

def dodaj_diagnostyke(
        obraz: np.ndarray,
        liczba_detekcji: int,
        prog_confidence: float,
) -> np.ndarray:
    """
    Dodaje napisy diagnostyczne w lewym górnym rogu obrazu:
    - liczba wykrytych obiektów,
    - aktualny próg confidence.
    """
    wynik = obraz.copy()

    linie = [
        f"Detekcje: {liczba_detekcji}",
        f"Prog conf: {prog_confidence:.2f}",
    ]

    y = 28
    for linia in linie:
        (tw, th), _ = cv2.getTextSize(linia, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        # Ciemne tło pod tekstem diagnostycznym
        cv2.rectangle(wynik, (6, y - th - 4), (6 + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(
            wynik, linia, (8, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        y += th + 12

    return wynik


# ============================================================
# TODO 11: Sprawdzenie struktury danych treningowych
# ============================================================

def sprawdz_strukture_danych(sciezka_danych: str) -> bool:
    """
    Sprawdza czy plik data.yaml istnieje.
    Ultralytics sam weryfikuje strukturę katalogów przy uruchomieniu treningu.
    """
    # Konwertuj na ścieżkę absolutną – bezpieczne na każdym systemie
    sciezka_abs = os.path.abspath(sciezka_danych)

    if not os.path.isfile(sciezka_abs):
        print(f"[BŁĄD] Nie znaleziono pliku: {sciezka_abs}")
        print(f"[INFO] Upewnij się, że podajesz poprawną ścieżkę do data.yaml")
        return False

    print(f"[OK] data.yaml znaleziony: {sciezka_abs}")
    return True


# ============================================================
# TODO 12: Informacja o anotacji danych
# ============================================================

def wypisz_informacje_o_anotacji():
    """
    Wyświetla instrukcję dotyczącą przygotowania danych treningowych w formacie YOLO.
    """
    print("Zalecane narzędzia do anotacji:")
    print("  - LabelImg, CVAT, Makesense.ai, Roboflow")
    print()
    print("Format anotacji YOLO:")
    print("  <class_id> <x_center> <y_center> <width> <height>")
    print("  Przykład: 0 0.523 0.441 0.210 0.317")
    print()
    print("Struktura datasetu:")
    print("  dataset/images/train/")
    print("  dataset/images/val/")
    print("  dataset/labels/train/")
    print("  dataset/labels/val/")
    print("  dataset/data.yaml")


# ============================================================
# TODO 13: Dotrenowanie modelu (fine-tuning)
# ============================================================

def dotrenuj_model(
        model,
        sciezka_danych: str,
        liczba_epok: int = 20,
        rozmiar_obrazu: int = 640,
):
    """
    Uruchamia fine-tuning modelu YOLO na własnym zbiorze danych.
    Przed treningiem sprawdza czy plik data.yaml istnieje.
    """
    print(f"\n[TRENING] Sprawdzanie pliku konfiguracyjnego...")
    if not sprawdz_strukture_danych(sciezka_danych):
        print("[BŁĄD] Popraw ścieżkę do data.yaml i spróbuj ponownie.")
        sys.exit(1)

    print(f"[TRENING] Uruchamianie treningu: epochs={liczba_epok}, imgsz={rozmiar_obrazu}")
    print(f"[TRENING] Dane: {sciezka_danych}\n")

    # Uruchom trening – ultralytics sam zweryfikuje strukturę katalogów
    model.train(
        data=sciezka_danych,
        epochs=liczba_epok,
        imgsz=rozmiar_obrazu,
    )

    # Ścieżka do najlepszych wag zapisanych przez Ultralytics
    najlepszy_model = os.path.join("runs", "detect", "train", "weights", "best.pt")
    print(f"\n[TRENING] Trening zakończony!")
    print(f"[TRENING] Najlepszy model zapisany w: {najlepszy_model}")
    return najlepszy_model


# ============================================================
# TODO 14: Wczytanie modelu dotrenowanego
# ============================================================

def wczytaj_model_dotrenowany(sciezka_modelu: str):
    """
    Wczytuje model zapisany po zakończeniu fine-tuningu (zwykle best.pt).
    """
    if not os.path.isfile(sciezka_modelu):
        print(f"[BŁĄD] Nie znaleziono pliku modelu dotrenowanego: '{sciezka_modelu}'")
        sys.exit(1)
    print(f"[OK] Model dotrenowany wczytany: {sciezka_modelu}")
    return YOLO(sciezka_modelu)


# ============================================================
# TODO 15: Porównanie modelu bazowego i dotrenowanego
# ============================================================

def porownaj_modele(
        model_bazowy,
        model_dotrenowany,
        obraz: np.ndarray,
        prog_confidence: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uruchamia oba modele na tym samym obrazie i zwraca dwa obrazy wynikowe
    do wyświetlenia w osobnych oknach.
    """
    nazwy_bazowe = pobierz_nazwy_klas(model_bazowy)
    nazwy_dotrenowane = pobierz_nazwy_klas(model_dotrenowany)

    # Detekcja modelem bazowym
    wyniki_baz = wykonaj_detekcje(model_bazowy, obraz)
    det_baz = filtruj_detekcje(odczytaj_wyniki(wyniki_baz), prog_confidence)
    obraz_baz = rysuj_detekcje(obraz, det_baz, nazwy_bazowe)
    obraz_baz = dodaj_diagnostyke(obraz_baz, len(det_baz), prog_confidence)

    # Detekcja modelem dotrenowanym
    wyniki_dot = wykonaj_detekcje(model_dotrenowany, obraz)
    det_dot = filtruj_detekcje(odczytaj_wyniki(wyniki_dot), prog_confidence)
    obraz_dot = rysuj_detekcje(obraz, det_dot, nazwy_dotrenowane)
    obraz_dot = dodaj_diagnostyke(obraz_dot, len(det_dot), prog_confidence)

    # Dodaj nagłówek tekstowy nad każdym obrazem
    def _dodaj_naglowek(img: np.ndarray, tekst: str) -> np.ndarray:
        h, w = img.shape[:2]
        banner = np.zeros((36, w, 3), dtype=np.uint8)
        cv2.putText(banner, tekst, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        return np.vstack([banner, img])

    obraz_baz = _dodaj_naglowek(obraz_baz, "Model bazowy")
    obraz_dot = _dodaj_naglowek(obraz_dot, "Model dotrenowany")

    return obraz_baz, obraz_dot


# ============================================================
# Przetwarzanie pojedynczego obrazu
# ============================================================

def przetwarzaj_obraz(model, sciezka_obrazu: str, prog_confidence: float):
    """
    Wczytuje obraz, wykonuje detekcję i wyświetla wyniki w oknie.
    """
    obraz = wczytaj_obraz(sciezka_obrazu)

    wyniki = wykonaj_detekcje(model, obraz)
    detekcje = odczytaj_wyniki(wyniki)
    detekcje = filtruj_detekcje(detekcje, prog_confidence)

    nazwy_klas = pobierz_nazwy_klas(model)

    obraz_wynikowy = rysuj_detekcje(obraz, detekcje, nazwy_klas)
    obraz_wynikowy = dodaj_diagnostyke(obraz_wynikowy, len(detekcje), prog_confidence)

    print(f"[INFO] Wykryto obiektów: {len(detekcje)}")
    cv2.imshow("Wynik detekcji - obraz", obraz_wynikowy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Przetwarzanie wideo lub obrazu z kamery
# ============================================================

def przetwarzaj_wideo(model, zrodlo: str, prog_confidence: float):
    """
    Przetwarza klatki ze strumienia wideo lub kamery w pętli.
    Naciśnij Q lub Esc, aby zakończyć.
    """
    cap = wczytaj_strumien(zrodlo)
    nazwy_klas = pobierz_nazwy_klas(model)

    print("[INFO] Naciśnij Q lub Esc, aby zakończyć.")

    while True:
        poprawnie, klatka = cap.read()
        if not poprawnie:
            # Koniec pliku lub utrata sygnału kamery
            break

        wyniki = wykonaj_detekcje(model, klatka)
        detekcje = odczytaj_wyniki(wyniki)
        detekcje = filtruj_detekcje(detekcje, prog_confidence)

        klatka_wynikowa = rysuj_detekcje(klatka, detekcje, nazwy_klas)
        klatka_wynikowa = dodaj_diagnostyke(klatka_wynikowa, len(detekcje), prog_confidence)

        cv2.imshow("Wynik detekcji - wideo", klatka_wynikowa)

        # Sprawdź czy użytkownik nacisnął Q lub Esc
        klawisz = cv2.waitKey(1) & 0xFF
        if klawisz in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Tryb porównawczy – dwa modele na jednym obrazie
# ============================================================

def uruchom_porownanie(
        model_bazowy,
        model_dotrenowany,
        sciezka_obrazu: str,
        prog_confidence: float,
):
    """
    Wyświetla wyniki detekcji obu modeli w osobnych oknach.
    """
    obraz = wczytaj_obraz(sciezka_obrazu)

    obraz_bazowy, obraz_dotrenowany = porownaj_modele(
        model_bazowy,
        model_dotrenowany,
        obraz,
        prog_confidence,
    )

    cv2.imshow("Model bazowy", obraz_bazowy)
    cv2.imshow("Model dotrenowany", obraz_dotrenowany)
    print("[INFO] Naciśnij dowolny klawisz, aby zakończyć.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# TODO 16–20: Funkcja główna
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LAB6 - YOLO: detekcja i fine-tuning")
    parser.add_argument("--model", required=True, help="Ścieżka do modelu bazowego, np. yolov8n.pt")
    parser.add_argument("--image", help="Ścieżka do obrazu")
    parser.add_argument("--video", help="Ścieżka do pliku wideo")
    parser.add_argument("--camera", action="store_true", help="Użyj kamery")
    parser.add_argument("--confidence", type=float, default=0.5, help="Próg confidence (domyślnie 0.5)")

    parser.add_argument("--train", action="store_true", help="Uruchom dotrenowanie modelu")
    parser.add_argument("--train-data", help="Ścieżka do pliku data.yaml, np. dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=20, help="Liczba epok (domyślnie 20)")
    parser.add_argument("--imgsz", type=int, default=640, help="Rozmiar obrazu do treningu (domyślnie 640)")
    parser.add_argument("--trained-model",
                        help="Ścieżka do modelu dotrenowanego, np. runs/detect/train/weights/best.pt")
    parser.add_argument("--compare", action="store_true", help="Porównaj model bazowy i dotrenowany")
    parser.add_argument("--show-annotation-help", action="store_true", help="Pokaż informacje o anotacji danych")

    args = parser.parse_args()

    # TODO 16: Wczytaj model bazowy podany przez użytkownika
    model = wczytaj_model(args.model)

    # TODO 17: Wyświetl informacje o anotacji danych i zakończ
    if args.show_annotation_help:
        wypisz_informacje_o_anotacji()
        return 0

    # TODO 18: Uruchom dotrenowanie modelu na własnym datasecie
    if args.train:
        if not args.train_data:
            print("[BŁĄD] Podaj ścieżkę do data.yaml: --train-data dataset/data.yaml")
            return 1
        dotrenuj_model(model, args.train_data, args.epochs, args.imgsz)
        return 0

    # TODO 19: Porównaj model bazowy z modelem dotrenowanym na jednym obrazie
    if args.compare:
        if not args.trained_model:
            print("[BŁĄD] Podaj ścieżkę do modelu dotrenowanego: --trained-model <ścieżka>")
            return 1
        if not args.image:
            print("[BŁĄD] Podaj ścieżkę do obrazu: --image <ścieżka>")
            return 1
        model_dot = wczytaj_model_dotrenowany(args.trained_model)
        uruchom_porownanie(model, model_dot, args.image, args.confidence)
        return 0

    # TODO 20: Obsłuż tryby: obraz, wideo, kamera
    if args.image:
        przetwarzaj_obraz(model, args.image, args.confidence)
    elif args.video:
        przetwarzaj_wideo(model, args.video, args.confidence)
    elif args.camera:
        przetwarzaj_wideo(model, "camera", args.confidence)
    else:
        print("[BŁĄD] Podaj tryb działania: --image, --video, --camera, --train lub --compare")
        print("       Pomoc: python lab6.py --help")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
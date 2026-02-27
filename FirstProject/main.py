from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import cv2
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')          # Rendu sans fenêtre popup (nécessaire avec PyQt5)
import matplotlib.pyplot as plt

# ── 2. CHARGEMENT DYNAMIQUE DE L'INTERFACE ──────────────────
qtcreator_file = "design.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


# ── 3. CLASSE PRINCIPALE ────────────────────────────────────
class DesignWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(DesignWindow, self).__init__()
        self.setupUi(self)          # Charge tous les widgets définis dans design.ui

        # Variable qui stockera l'image OpenCV chargée (BGR)
        self.current_image = None

        # ── 4. CONNEXION DES BOUTONS AUX FONCTIONS ──────────
        self.Browse.clicked.connect(self.get_image)

        self.DisplayRedChan.clicked.connect(self.showRedChannel)
        self.DisplayGreenChan.clicked.connect(self.showGreenChannel)
        self.DisplayBlueChan.clicked.connect(self.showBlueChannel)

        self.DisplayColorHist.clicked.connect(self.show_HistColor)

        self.DisplayGrayImg.clicked.connect(self.show_UpdatedImgGray)
        self.DisplayGrayHist.clicked.connect(self.show_HistGray)

    # ════════════════════════════════════════════════════════
    #  FONCTIONS UTILITAIRES
    # ════════════════════════════════════════════════════════

    def convert_cv_qt(self, cv_image):
        """
        Convertit une image OpenCV (numpy array BGR ou Grayscale)
        en QPixmap pour l'afficher dans un QLabel PyQt5.
        """
        if len(cv_image.shape) == 2:
            # Image en niveaux de gris : on la convertit en BGR pour QImage
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

        h, w, ch = cv_image.shape
        bytes_per_line = ch * w
        # Format_BGR888 correspond exactement au format OpenCV (Blue, Green, Red)
        cv_image_Qt = QtGui.QImage(
            cv_image.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888
        )
        return QPixmap.fromImage(cv_image_Qt)

    def display_image_in_label(self, label_widget, cv_image):
        """
        Redimensionne et affiche une image OpenCV dans un QLabel
        en conservant les proportions.
        """
        pixmap = self.convert_cv_qt(cv_image)
        # Ajuste la pixmap à la taille du label sans déformer l'image
        scaled = pixmap.scaled(
            label_widget.width(),
            label_widget.height(),
            aspectRatioMode=QtCore.Qt.KeepAspectRatio
        )

        label_widget.setPixmap(scaled)

    def makeFigure(self, label_widget, fig_path):
        """
        Charge une figure (enregistrée sur disque comme PNG)
        et l'affiche dans le QLabel cible.
        """
        pixmap = QPixmap(fig_path)
        scaled = pixmap.scaled(
            label_widget.width(),
            label_widget.height(),
            aspectRatioMode=QtCore.Qt.KeepAspectRatio
        )
        label_widget.setPixmap(scaled)

    def showDimensions(self):
        """
        Affiche les dimensions (hauteur, largeur, canaux) de l'image
        dans le QLabel 'Dimensions'.
        """
        if self.current_image is None:
            return

        if len(self.current_image.shape) == 3:
            h, w, ch = self.current_image.shape
        else:
            h, w = self.current_image.shape
            ch = 1   # Image en niveaux de gris = 1 canal

        self.Dimensions.setText(
            f"Hauteur: {h}\nLargeur: {w}\nNombre de canaux: {ch}"
        )

    # ════════════════════════════════════════════════════════
    #  2.2 – CHARGEMENT DE L'IMAGE
    # ════════════════════════════════════════════════════════

    def get_image(self):
        """
        Ouvre un explorateur de fichiers, charge l'image sélectionnée
        avec OpenCV, l'affiche dans 'OriginalImg' et affiche ses dimensions.
        """
        # Ouvre la boîte de dialogue pour choisir un fichier image
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner une image",
            "",
            "Images (*.jpg *.jpeg *.png)"
        )

        if not file_path:
            return   # L'utilisateur a annulé

        # Lecture de l'image avec OpenCV (format BGR par défaut)
        self.current_image = cv2.imread(file_path)

        if self.current_image is None:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Impossible de lire l'image.")
            return

        # Affichage de l'image originale dans le widget dédié
        self.display_image_in_label(self.OriginalImg, self.current_image)

        # Affichage des dimensions
        self.showDimensions()

    # ════════════════════════════════════════════════════════
    #  2.4 – EXTRACTION ET AFFICHAGE DES CANAUX
    # ════════════════════════════════════════════════════════

    def showRedChannel(self):
        """
        Extrait le canal Rouge de l'image et l'affiche en couleur dans RedChannel.
        Les autres canaux sont mis à zéro pour ne montrer que le rouge.
        """
        if self.current_image is None:
            return

        # Crée une image noire de la même taille
        red_img = np.zeros_like(self.current_image)
        # OpenCV stocke l'image en BGR : canal 2 = Rouge
        red_img[:, :, 2] = self.current_image[:, :, 2]

        self.display_image_in_label(self.RedChannel, red_img)

    def showGreenChannel(self):
        """
        Extrait le canal Vert et l'affiche en couleur dans GreenChannel.
        """
        if self.current_image is None:
            return

        green_img = np.zeros_like(self.current_image)
        # Canal 1 = Vert en BGR
        green_img[:, :, 1] = self.current_image[:, :, 1]

        self.display_image_in_label(self.GreenChannel, green_img)

    def showBlueChannel(self):
        """
        Extrait le canal Bleu et l'affiche en couleur dans BlueChannel.
        """
        if self.current_image is None:
            return

        blue_img = np.zeros_like(self.current_image)
        # Canal 0 = Bleu en BGR
        blue_img[:, :, 0] = self.current_image[:, :, 0]

        self.display_image_in_label(self.BlueChannel, blue_img)

    # ════════════════════════════════════════════════════════
    #  2.5 – HISTOGRAMME COULEUR
    # ════════════════════════════════════════════════════════

    def show_HistColor(self):
        """
        Calcule et affiche l'histogramme des 3 canaux (R, V, B) de l'image couleur.
        Enregistre la figure sous 'Color_Histogram.png' puis l'affiche dans ColorHist.
        """
        if self.current_image is None:
            return

        fig, ax = plt.subplots(figsize=(6, 3))

        # On trace un histogramme par canal avec sa couleur correspondante
        colors_bgr = ('b', 'g', 'r')   # Ordre BGR d'OpenCV
        labels     = ('Bleu', 'Vert', 'Rouge')

        for i, (col, lbl) in enumerate(zip(colors_bgr, labels)):
            hist = cv2.calcHist([self.current_image], [i], None, [256], [0, 256])
            ax.plot(hist, color=col, label=lbl)

        ax.set_title("Histogramme Couleur")
        ax.set_xlabel("Intensité (0-255)")
        ax.set_ylabel("Nombre de pixels")
        ax.legend()
        fig.tight_layout()

        # Enregistrement de la figure en tant qu'image PNG
        fig.savefig("Color_Histogram.png")
        plt.close(fig)

        # Affichage dans le widget ColorHist
        self.makeFigure(self.ColorHist, "Color_Histogram.png")

    # ════════════════════════════════════════════════════════
    #  2.6 – CONTRASTE ET BRILLANCE
    # ════════════════════════════════════════════════════════

    def getContrast(self):
        """
        Lit et retourne la valeur du contraste (alpha) depuis le champ 'Contrast'.
        Valeur par défaut : 1.0 (pas de modification)
        alpha > 1 → augmente le contraste
        alpha < 1 → diminue le contraste
        """
        try:
            return float(self.Contrast.text())
        except ValueError:
            return 1.0   # Valeur par défaut si la saisie est invalide

    def getBrightness(self):
        """
        Lit et retourne la valeur de brillance (beta) depuis le champ 'Brightness'.
        Valeur par défaut : 0 (pas de modification)
        beta > 0 → augmente la luminosité
        beta < 0 → diminue la luminosité
        """
        try:
            return int(self.Brightness.text())
        except ValueError:
            return 0   # Valeur par défaut si la saisie est invalide

    # ════════════════════════════════════════════════════════
    #  2.7 – IMAGE EN NIVEAUX DE GRIS AVEC CONTRASTE/BRILLANCE
    # ════════════════════════════════════════════════════════

    def show_UpdatedImgGray(self):
        """
        Convertit l'image en niveaux de gris, applique le contraste et la brillance
        lus dans les champs de saisie, puis affiche le résultat dans GrayImg.

        Formule : pixel_sortie = alpha * pixel_entrée + beta
        cv2.convertScaleAbs gère automatiquement le clamp dans [0, 255].
        """
        if self.current_image is None:
            return

        # Conversion BGR → Niveaux de gris : Y = 0.299R + 0.587G + 0.114B
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

        # Application du contraste (alpha) et de la brillance (beta)
        alpha = self.getContrast()
        beta  = self.getBrightness()
        updated_gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # On sauvegarde l'image grise modifiée pour l'histogramme
        self.current_gray = updated_gray

        # Affichage dans le widget GrayImg
        self.display_image_in_label(self.GrayImg, updated_gray)

    # ════════════════════════════════════════════════════════
    #  2.8 – HISTOGRAMME EN NIVEAUX DE GRIS
    # ════════════════════════════════════════════════════════

    def calc_HistGray(self):
        """
        Calcule l'histogramme de l'image en niveaux de gris (après modification
        du contraste et de la brillance).
        Retourne le tableau d'histogramme ou None si aucune image n'est chargée.
        """
        if self.current_image is None:
            return None

        # Si l'image grise modifiée n'existe pas encore, on la crée
        if not hasattr(self, 'current_gray') or self.current_gray is None:
            gray  = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            alpha = self.getContrast()
            beta  = self.getBrightness()
            self.current_gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # cv2.calcHist : [image], [canal], masque, [nb_bins], [plage]
        hist = cv2.calcHist([self.current_gray], [0], None, [256], [0, 256])
        return hist

    def show_HistGray(self):
        """
        Calcule l'histogramme niveaux de gris, enregistre la figure sous
        'Gray_Histogram.png' et l'affiche dans le widget GrayHist.
        """
        hist = self.calc_HistGray()
        if hist is None:
            return

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(hist, color='gray')
        ax.set_title("Histogramme Niveaux de Gris")
        ax.set_xlabel("Intensité (0-255)")
        ax.set_ylabel("Nombre de pixels")
        ax.fill_between(range(256), hist.flatten(), alpha=0.3, color='gray')
        fig.tight_layout()

        # Enregistrement de la figure
        fig.savefig("Gray_Histogram.png")
        plt.close(fig)

        # Affichage dans le widget GrayHist
        self.makeFigure(self.GrayHist, "Gray_Histogram.png")


# ── 5. FONCTION PRINCIPALE (ENTRY POINT) ────────────────────
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DesignWindow()
    window.show()
    sys.exit(app.exec_())

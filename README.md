# flipatron
Allows you to scan pages and books using just a video of you flipping through the pages

## opencv install
use make build
PKGBUILD - https://stackoverflow.com/questions/15320267/package-opencv-was-not-found-in-the-pkg-config-search-path

## SAM model
To use any of the files or notebooks utilizing the slower Meta SAM model, install from requirements.txt
```console
conda activate <env-name>
pip install -r requirements.txt
```

## FastSAM model
To use any of the files or notebooks utilizing the FastSAM model, install from fast_requirements.txt

```console
conda activate <env-name>
pip install -r fast_requirements.txt
```

## Flutter
```console
cd app/
flutter pub get
flutter run
```

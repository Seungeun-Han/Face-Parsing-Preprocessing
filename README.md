# Face Parsing Preprocessing

Face Parsing Task를 수행하기 위한 전처리 알고리즘 코드입니다.

모든 코드는 직접 작성되었습니다.

<hr>

## 코드 설명
코드의 기능, 입력, 출력을 위주로 간단히 설명드리겠습니다.

코드 내의 모든 경로는 절대 경로로 작성되어있으며, 필요 시 "path" 변수를 변경하시면 됩니다.

더욱 자세한 설명은 코드 내 주석을 참고하시길 바랍니다.

<br>

<hr>

### [0or1_check.py](https://github.com/Seungeun-Han/Face-Parsing-Preprocessing/blob/main/0or1_check.py)

This code is designed to verify whether all pixel values in the image are 0~1 or not.

<br>

<hr>

### [Helen_align.py](https://github.com/Seungeun-Han/Face-Parsing-Preprocessing/tree/main)
Helen Dataset을 Face Align 하는 코드입니다.

Helen Dataset은 face가 align이 되어있지 않고, size가 다른 데이터셋입니다.

이 데이터셋을 그대로 학습에 넣을 수 있지만, 더 효과적인 학습을 위해 align 한 뒤의 얼굴을 넣을 수도 있습니다.

Helen Dataset은 각 이미지별 landmark 정보가 있는 landmark_txt.txt 파일을 제공합니다.

여기에서 왼/오른쪽 눈, 코, 왼/오른쪽 입 끝 부분에 대한 위치는 각각 105, 106, 55, 85, 101 번째 라인에 저장되어있습니다.

이를 저희가 설정한 포인트들로 align하고, 원하는 크기로 자르면 최종적으로 Face Align이 완료됩니다.

#### Helen Dataset Download
http://www.ifp.illinois.edu/~vuongle2/helen/

#### Example
왼/오른쪽 눈, 코, 왼/오른쪽 입 끝 부분을 [[182, 229], [295, 229], [238, 301], [190, 349], [288, 349]]로 옮기고, (473x473) 사이즈로 자른 예시입니다.

- Input Image

![30427236_2](https://github.com/Seungeun-Han/Face-Parsing-Preprocessing/assets/101082685/e4ba2bfb-b379-408a-bbfc-e9740e1fcd0a)

- Output Image

![30427236_2](https://github.com/Seungeun-Han/Face-Parsing-Preprocessing/assets/101082685/06333639-ba65-488b-93d4-090e875e61b7)

<br>

<hr>

### [LaPa_align.py](https://github.com/Seungeun-Han/Face-Parsing-Preprocessing/blob/main/LaPa_align.py)
위의 경우와 동일합니다.

#### LaPa Dataset Download
무슨 이유인지는 모르겠으나 공식 홈페이지에서 다운로드 링크를 내렸다(..)

#### Example
왼/오른쪽 눈, 코, 왼/오른쪽 입 끝 부분을 [[182, 229], [295, 229], [238, 301], [190, 349], [288, 349]]로 옮기고, (473x473) 사이즈로 자른 예시입니다.

- Input Image

![3013103_0](https://github.com/Seungeun-Han/Face-Parsing-Preprocessing/assets/101082685/9ebadc75-1337-45d2-8dbb-7bde0eec0855)

- Output Image

![3013103_0](https://github.com/Seungeun-Han/Face-Parsing-Preprocessing/assets/101082685/74381604-08f5-42c6-bc47-bc76b1046f6d)

<br>

<hr>


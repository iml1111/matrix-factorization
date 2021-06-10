# Matrix-Factorization
![image](https://user-images.githubusercontent.com/29897277/121314162-0db62f00-c942-11eb-81eb-bd41780028e8.png)

**Simple matrix factorization for Python**
해당 repo는 **matrix-factorization**으로 **movielens 데이터**를 학습하여 예측해보는 추천 시스템 코드 예제입니다.



## Dependency

- python 3.7 +

```shell
$ pip install -r requirements.txt
```



## Get Started

```shell
$ python train.py 
usage: train.py [-h] [--k K] [--n_epochs N_EPOCHS] [--lr LR] [--beta BETA]
                [--svd] [--sgd]

optional arguments:
  -h, --help           show this help message and exit
  --k K                latent factor size.
  --n_epochs N_EPOCHS  num of Iterations
  --lr LR              learning rate.
  --beta BETA          regularization parameter.
  --svd                Use SVD Algorithm.
  --sgd                Use SGD Algorithm.
```

### Train with SVD

```shell
$ python train.py --svd --k 300
```

### Train with SGD

```shell
$ python train.py --sgd --k 300 --lr 0.01 --beta 0.01 --n_epochs 100
```



## Algorithms

MF는 다음과 같은 알고리즘으로 구현되어 있습니다.

- SVD - Singular Value Decomposition
- SGD - Stochastic Gradient Descent

### SVD

![image](https://user-images.githubusercontent.com/29897277/121314621-7dc4b500-c942-11eb-88f7-95cc7e42983f.png)

특이값 분해(SVD) 알고리즘은, User와 Item 간의 평가 지표를 기록한 Rating Matrix를 위 규격에 맞게 분해한 후, 

분해된 각 matrix를 입력받은 하이퍼 파라미터 K를 따라 절단된 SVD(Truncated SVD) 형태로 행렬을 잘라냅니다.

그 후, 다시 Rating Matrix의 Shape에 맞춰 행렬을 복원해줌으로써, predict matrix를 생성합니다.

K를 크게 잡으면 기존의 Rating Matrix로부터 다양한 의미를 가져갈 수 있지만, K를 작게 잡아야만 핵심적인 정보외의 노이즈를 제거할 수 있습니다.

**matrix를 절단시키는 것으로, 설명력이 낮은 정보를 삭제하고 설명력이 높은 정보를 남긴다는 방향성으로 평가를 예측**하게 됩니다.



### SGD

![image](https://user-images.githubusercontent.com/29897277/121314509-608fe680-c942-11eb-8524-3f442f8148f2.png)

![image](https://user-images.githubusercontent.com/29897277/121320693-596bd700-c948-11eb-92bc-418e51b2f805.png)

User Latent Matrix와 Item Latent Matrix를 내적을 통해, 기존의 Rating Matrix를 유사하게 재현하는 알고리즘입니다. 사용된 최적화 알고리즘은 SGD(Stochastic Gradient Descent)이며, 매 epoch마다 Latent Matrix를 통해 Predict Matrix를 기반으로 기존의 Rating Matrix와의 Loss 값으로 최적화를 수행합니다. 



## Evaluation

### Used Dataset(MovieLens)

| Feature      | Value   |
| ------------ | ------- |
| type         | Small   |
| movies       | 9,000   |
| users        | 600     |
| ratings      | 100,000 |
| Last Updated | 2018-09 |

### Test Inferences

| parameters     | SVD           | SGD          |
| -------------- | ------------- | ------------ |
| K              | 300           | 300          |
| learning rate  | -             | 0.01         |
| beta           | -             | 0.001        |
| n_epochs       | -             | 300          |
| **Train RMSE** | **0.1796847** | **0.060330** |
| **Test RMSE**  | **3.0905050** | **0.759681** |





## References

- https://github.com/jaewonlee-728/fastcampus-RecSys
- https://velog.io/@vvakki_/Matrix-Factorization-2

# Training Cifar10 in Python using Resnet50 network


* Results After 20 Epoch Trained Model : ~%80
```
./trained_models/resnet50model1.h5 is loaded!
1000/1000 [==============================] - 3s 3ms/step
Wrong class number:
209/1000 is wrongly classified
```


* Training Results With 20 Epoch

```
Total params: 8,599,434
Trainable params: 8,568,842
Non-trainable params: 30,592
__________________________________________________________________________________________________
incarnation: 6429009583015944562
physical_device_desc: "device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7"
]
Train on 50000 samples, validate on 10000 samples
Epoch 1/20
50000/50000 [==============================] - 175s 4ms/step - loss: 0.2956 - acc: 0.8990 - val_loss: 0.1936 - val_acc: 0.9258

Epoch 00001: val_acc improved from -inf to 0.92577, saving model to ./trained_models/model1_-01-0.93_2018-12-09_20.37.49.h5
Epoch 2/20
50000/50000 [==============================] - 167s 3ms/step - loss: 0.2092 - acc: 0.9209 - val_loss: 0.1573 - val_acc: 0.9384

Epoch 00002: val_acc improved from 0.92577 to 0.93837, saving model to ./trained_models/model1_-02-0.94_2018-12-09_20.37.49.h5
Epoch 3/20
50000/50000 [==============================] - 164s 3ms/step - loss: 0.1780 - acc: 0.9323 - val_loss: 0.1403 - val_acc: 0.9446

Epoch 00003: val_acc improved from 0.93837 to 0.94464, saving model to ./trained_models/model1_-03-0.94_2018-12-09_20.37.49.h5
Epoch 4/20
50000/50000 [==============================] - 154s 3ms/step - loss: 0.1579 - acc: 0.9394 - val_loss: 0.1298 - val_acc: 0.9486

Epoch 00004: val_acc improved from 0.94464 to 0.94861, saving model to ./trained_models/model1_-04-0.95_2018-12-09_20.37.49.h5
Epoch 5/20
50000/50000 [==============================] - 158s 3ms/step - loss: 0.1439 - acc: 0.9446 - val_loss: 0.1228 - val_acc: 0.9515

Epoch 00005: val_acc improved from 0.94861 to 0.95148, saving model to ./trained_models/model1_-05-0.95_2018-12-09_20.37.49.h5
Epoch 6/20
50000/50000 [==============================] - 153s 3ms/step - loss: 0.1324 - acc: 0.9491 - val_loss: 0.1171 - val_acc: 0.9537

Epoch 00006: val_acc improved from 0.95148 to 0.95374, saving model to ./trained_models/model1_-06-0.95_2018-12-09_20.37.49.h5
Epoch 7/20
50000/50000 [==============================] - 160s 3ms/step - loss: 0.1235 - acc: 0.9527 - val_loss: 0.1128 - val_acc: 0.9557

Epoch 00007: val_acc improved from 0.95374 to 0.95571, saving model to ./trained_models/model1_-07-0.96_2018-12-09_20.37.49.h5
Epoch 8/20
50000/50000 [==============================] - 155s 3ms/step - loss: 0.1143 - acc: 0.9561 - val_loss: 0.1107 - val_acc: 0.9566

Epoch 00008: val_acc improved from 0.95571 to 0.95659, saving model to ./trained_models/model1_-08-0.96_2018-12-09_20.37.49.h5
Epoch 9/20
50000/50000 [==============================] - 163s 3ms/step - loss: 0.1079 - acc: 0.9588 - val_loss: 0.1078 - val_acc: 0.9583

Epoch 00009: val_acc improved from 0.95659 to 0.95829, saving model to ./trained_models/model1_-09-0.96_2018-12-09_20.37.49.h5
Epoch 10/20
50000/50000 [==============================] - 164s 3ms/step - loss: 0.0996 - acc: 0.9617 - val_loss: 0.1060 - val_acc: 0.9589

Epoch 00010: val_acc improved from 0.95829 to 0.95888, saving model to ./trained_models/model1_-10-0.96_2018-12-09_20.37.49.h5
Epoch 11/20
50000/50000 [==============================] - 158s 3ms/step - loss: 0.0948 - acc: 0.9634 - val_loss: 0.1041 - val_acc: 0.9603

Epoch 00011: val_acc improved from 0.95888 to 0.96030, saving model to ./trained_models/model1_-11-0.96_2018-12-09_20.37.49.h5
Epoch 12/20
50000/50000 [==============================] - 165s 3ms/step - loss: 0.0884 - acc: 0.9662 - val_loss: 0.1041 - val_acc: 0.9605

Epoch 00012: val_acc improved from 0.96030 to 0.96046, saving model to ./trained_models/model1_-12-0.96_2018-12-09_20.37.49.h5
Epoch 13/20
50000/50000 [==============================] - 163s 3ms/step - loss: 0.0828 - acc: 0.9686 - val_loss: 0.1032 - val_acc: 0.9609

Epoch 00013: val_acc improved from 0.96046 to 0.96087, saving model to ./trained_models/model1_-13-0.96_2018-12-09_20.37.49.h5
Epoch 14/20
50000/50000 [==============================] - 154s 3ms/step - loss: 0.0766 - acc: 0.9709 - val_loss: 0.1028 - val_acc: 0.9616

Epoch 00014: val_acc improved from 0.96087 to 0.96162, saving model to ./trained_models/model1_-14-0.96_2018-12-09_20.37.49.h5
Epoch 15/20
50000/50000 [==============================] - 161s 3ms/step - loss: 0.0716 - acc: 0.9728 - val_loss: 0.1042 - val_acc: 0.9612

Epoch 00015: val_acc did not improve from 0.96162
Epoch 16/20
50000/50000 [==============================] - 164s 3ms/step - loss: 0.0656 - acc: 0.9752 - val_loss: 0.1048 - val_acc: 0.9617

Epoch 00016: val_acc improved from 0.96162 to 0.96170, saving model to ./trained_models/model1_-16-0.96_2018-12-09_20.37.49.h5
Epoch 17/20
50000/50000 [==============================] - 159s 3ms/step - loss: 0.0616 - acc: 0.9767 - val_loss: 0.1060 - val_acc: 0.9617

Epoch 00017: val_acc did not improve from 0.96170
Epoch 18/20
50000/50000 [==============================] - 153s 3ms/step - loss: 0.0567 - acc: 0.9787 - val_loss: 0.1070 - val_acc: 0.9622

Epoch 00018: val_acc improved from 0.96170 to 0.96216, saving model to ./trained_models/model1_-18-0.96_2018-12-09_20.37.49.h5
Epoch 19/20
50000/50000 [==============================] - 159s 3ms/step - loss: 0.0526 - acc: 0.9804 - val_loss: 0.1092 - val_acc: 0.9619

Epoch 00019: val_acc did not improve from 0.96216
Epoch 20/20
50000/50000 [==============================] - 164s 3ms/step - loss: 0.0475 - acc: 0.9821 - val_loss: 0.1107 - val_acc: 0.9618

Epoch 00020: val_acc did not improve from 0.96216
Training time: 3243.37100315094
10000/10000 [==============================] - 6s 605us/step
[INFO] loss=0.1107, accuracy: 96.1800%
model1 resaved.
```
<details>
  <summary>Layers of the Model. Warning, big image inside!</summary>
 <img src="outputs/model1_plot.png">
</details>

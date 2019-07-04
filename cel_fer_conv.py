import tensorflow as tf
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as pt

cel=np.array([-40,-10,0,8,15,22,38],dtype=float)
fer=np.array([-40,13,32,46,59,72,100],dtype=float)

lay0=tf.keras.layers.Dense(units=1,input_shape=[1])
model=tf.keras.Sequential([lay0])
model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.001))

hstory=model.fit(cel,fer,epochs=500,verbose=False) #epoch is the flag for full iteraton
print("Trained")

pt.xlabel('epoch')
pt.ylabel('loss magni')
pt.plot(hstory.history['loss'])

print(model.predict([100.0]))

This project is missing the model file included in the object detection lab from the SSD model "model.ckpt.data-00000-of-00001" because it's too big for Github.

Currently the image classifier is turned off since it's not working sufficently.
I did attempt the SSD model in the lab as the basis for detecting traffic lights and then imported the traffic light detection from the previous class and tested this on various images. I realized i needed another layer of classification however which was detecting lights facing your own lane as the output from 3 available models in the lab was bounded images of traffic lights pointing in any direction, which is not what we want.
In addition, the simulator was taking much too long to process the images which i suspect is due to not running tensorflow correctly and inefficient repeated memory transfer may be used? 

I realize I could try some other approaches like training a brand new classifier just on traffic lights using the simulator but it's unclear to me if this simulation data will be broad enough to make the model generic and able to handle all the variations (like rain on camera, sunset and haze conditions).
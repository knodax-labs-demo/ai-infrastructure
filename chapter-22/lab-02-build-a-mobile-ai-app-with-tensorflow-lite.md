# Hands-On Lab: Build a Mobile AI App with TensorFlow Lite

In this lab, you will build a simple Android application that performs **image classification entirely on the mobile device using TensorFlow Lite**.

The application will package a MobileNet TensorFlow Lite model with the Android app, capture or receive image input, preprocess the image locally, execute inference on the device, and display the predicted class without sending the image to a cloud inference service.

You will also explore optional GPU acceleration and compare execution performance on the actual target device.

The complete mobile AI workflow is:

```text
Camera / Image
      ↓
Image Preprocessing
      ↓
TensorFlow Lite Model
      ↓
On-Device Inference
      ↓
Class Probabilities
      ↓
Top Prediction
      ↓
Android UI
```

---

## Lab Objective

By completing this lab, you will learn how to:

* Integrate a TensorFlow Lite model into an Android application.
* Package a MobileNet model with the application.
* Load a `.tflite` model from Android assets.
* Prepare camera input for inference.
* Execute image classification locally.
* Convert model output into human-readable labels.
* Display predictions in the Android UI.
* Understand the architecture of on-device AI.
* Explore optional GPU acceleration.
* Measure inference performance on mobile hardware.

---

## Estimated Time

**Approximately 90–150 minutes**

---

## Tools

This lab uses:

* Android Studio
* Kotlin
* Android SDK
* TensorFlow Lite
* TensorFlow Lite Support libraries
* Optional TensorFlow Lite GPU support
* MobileNet model
* Android CameraX
* Android physical device or emulator

For camera-based testing, a physical Android device is generally preferable.

---

# Step 1: Verify the Prerequisites

Before starting, ensure you have:

* Android Studio installed.
* An Android device or emulator.
* A TensorFlow Lite MobileNet model.
* A matching `labels.txt` file.
* Basic familiarity with Kotlin.
* Basic familiarity with Android development.

For example:

```text
mobilenet_v2.tflite
labels.txt
```

The model and labels must correspond to one another.

The basic deployment architecture is:

```text
Android Application
       │
       ├── TensorFlow Lite Model
       ├── Labels
       ├── Camera Input
       └── Kotlin Inference Code
```

---

# Step 2: Create the Android Studio Project

Open Android Studio and create a new project.

Choose an:

```text
Empty Activity
```

Configure the project with settings similar to:

```text
Package:
com.example.tfliteclassifier

Language:
Kotlin

Minimum SDK:
API 23 or later
```

Allow the initial Gradle synchronization to complete before adding TensorFlow Lite components.

---

# Step 3: Understand the Project Structure

A simplified Android project might look like:

```text
app/
├── src/
│   └── main/
│       ├── java/
│       │   └── com/example/tfliteclassifier/
│       │       ├── MainActivity.kt
│       │       └── Classifier.kt
│       │
│       ├── assets/
│       │   ├── mobilenet_v2.tflite
│       │   └── labels.txt
│       │
│       ├── res/
│       └── AndroidManifest.xml
│
└── build.gradle
```

For the book companion repository, I recommend keeping only the most relevant source files:

```text
chapter-22/
├── lab-11-build-a-mobile-ai-app-with-tensorflow-lite.md
├── lab-11-classifier.kt
└── lab-11-main-activity.kt
```

---

# Step 4: Add TensorFlow Lite Dependencies

Open the application module's Gradle configuration.

Add TensorFlow Lite dependencies appropriate for your Android project.

For example:

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:<version>'
    implementation 'org.tensorflow:tensorflow-lite-support:<version>'
    implementation 'org.tensorflow:tensorflow-lite-gpu:<version>'
}
```

Replace:

```text
<version>
```

with mutually compatible versions appropriate for the project.

The GPU dependency is optional.

If you do not plan to test GPU acceleration, it does not need to be included.

---

## Dependency Relationship

Conceptually:

```text
Android Application
       ↓
TensorFlow Lite Runtime
       ↓
Model Execution
```

Optional acceleration adds:

```text
TensorFlow Lite
       ↓
GPU Delegate
       ↓
Mobile GPU
```

After modifying Gradle, synchronize the project.

---

# Step 5: Add the Model and Label Files

Create the assets directory if necessary:

```text
app/src/main/assets/
```

Place the model at:

```text
app/src/main/assets/mobilenet_v2.tflite
```

Place the label file at:

```text
app/src/main/assets/labels.txt
```

The project now contains:

```text
assets/
├── mobilenet_v2.tflite
└── labels.txt
```

Because the model is packaged with the application, it can run without downloading the model from a remote server.

---

# Step 6: Understand On-Device AI

The architecture is fundamentally different from cloud inference.

## Cloud Inference

```text
Camera
   ↓
Mobile Device
   ↓
Internet
   ↓
Cloud API
   ↓
Model
   ↓
Prediction
   ↓
Mobile Device
```

## On-Device Inference

```text
Camera
   ↓
Mobile Device
   ↓
TensorFlow Lite
   ↓
Prediction
```

The image can remain entirely on the device.

---

# Step 7: Create the Classifier

Create:

```text
Classifier.kt
```

Add:

```kotlin
import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class Classifier(
    private val assetManager: AssetManager
) {

    private val interpreter: Interpreter

    init {
        interpreter = Interpreter(
            loadModel("mobilenet_v2.tflite")
        )
    }

    private fun loadModel(
        modelName: String
    ): MappedByteBuffer {

        val fileDescriptor: AssetFileDescriptor =
            assetManager.openFd(modelName)

        FileInputStream(
            fileDescriptor.fileDescriptor
        ).use { inputStream ->

            val fileChannel =
                inputStream.channel

            val startOffset =
                fileDescriptor.startOffset

            val declaredLength =
                fileDescriptor.declaredLength

            return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                startOffset,
                declaredLength
            )
        }
    }

    fun runInference(
        input: Array<Array<Array<FloatArray>>>
    ): FloatArray {

        val output =
            Array(1) {
                FloatArray(1001)
            }

        interpreter.run(
            input,
            output
        )

        return output[0]
    }

    fun close() {
        interpreter.close()
    }
}
```

---

# Step 8: Understand Model Loading

The application loads:

```text
mobilenet_v2.tflite
```

from Android assets.

The process is:

```text
APK Assets
    ↓
Model File
    ↓
Memory Mapping
    ↓
TensorFlow Lite Interpreter
    ↓
Ready for Inference
```

Memory mapping allows the runtime to access the model efficiently without unnecessarily copying the entire file into application-managed memory.

---

# Step 9: Inspect the Model Tensor Requirements

Do not assume every MobileNet model uses identical tensors.

Model variations may differ in:

* Input shape
* Input type
* Output shape
* Number of labels
* Normalization
* Quantization
* RGB ordering

A common image input might be:

```text
1 × 224 × 224 × 3
```

but the actual `.tflite` model should be treated as authoritative.

The critical rule is:

```text
Application Preprocessing
          =
Model Training / Export Preprocessing
```

If these do not match, inference results may be incorrect even though the application runs successfully.

---

# Step 10: Load the Labels

The `labels.txt` file should contain the class names corresponding to the model output.

Conceptually:

```text
Output Index 0
      ↓
Label 0

Output Index 1
      ↓
Label 1

Output Index N
      ↓
Label N
```

A simple Kotlin approach can read the asset into a list:

```kotlin
val labels =
    assets.open("labels.txt")
        .bufferedReader()
        .readLines()
```

The label count should correspond to the model's output tensor.

---

# Step 11: Connect the Camera

Use Android's **CameraX** API to obtain images or video frames from the device camera.

The high-level pipeline is:

```text
CameraX
   ↓
Image Frame
   ↓
Bitmap / Image Buffer
   ↓
Resize
   ↓
RGB Conversion
   ↓
Normalization
   ↓
TensorFlow Lite Input
```

For a MobileNet model expecting:

```text
224 × 224 RGB
```

resize incoming images accordingly.

---

# Step 12: Preprocess the Image

Preprocessing is model-specific.

For a Float32 model, the pipeline might be:

```text
Camera Image
     ↓
Resize to 224 × 224
     ↓
Extract RGB Values
     ↓
Convert to Float
     ↓
Normalize
     ↓
Input Tensor
```

For a quantized model:

```text
Camera Image
     ↓
Resize
     ↓
Convert to Expected Integer Type
     ↓
Apply Quantization-Compatible Processing
     ↓
Input Tensor
```

Do not apply Float32 preprocessing blindly to a quantized model.

---

# Step 13: Call the Classifier

Once preprocessing produces the correct tensor, call:

```kotlin
classifier.runInference(
    preprocessedImage
)
```

The application flow becomes:

```text
Camera
   ↓
Preprocessing
   ↓
runInference()
   ↓
TFLite Interpreter
   ↓
Probability Array
```

---

# Step 14: Determine the Top Prediction

After inference:

```kotlin
val probabilities =
    classifier.runInference(
        preprocessedImage
    )
```

Find the highest-scoring output:

```kotlin
val topIndex =
    probabilities.indices.maxByOrNull {
        probabilities[it]
    } ?: -1
```

Then map it to a label:

```kotlin
if (
    topIndex >= 0 &&
    topIndex < labels.size
) {

    val label =
        labels[topIndex]

    textView.text =
        "Prediction: $label"
}
```

---

# Step 15: Understand Output Mapping

The model produces numbers:

```text
[0.001, 0.003, 0.91, 0.006, ...]
```

The application finds:

```text
Maximum Probability
       ↓
Output Index
       ↓
Label Lookup
       ↓
Human-Readable Prediction
```

For example:

```text
Index 281
    ↓
"tabby cat"
```

depending on the label set used by the model.

---

# Step 16: Complete Inference Path

The end-to-end mobile inference architecture is:

```text
Camera
   ↓
CameraX
   ↓
Image Frame
   ↓
Resize / Normalize
   ↓
Tensor Input
   ↓
TensorFlow Lite Interpreter
   ↓
MobileNet
   ↓
Probability Vector
   ↓
Highest Probability
   ↓
Label Lookup
   ↓
Android UI
```

This complete process remains on the mobile device.

---

# Step 17: Run the Application

Build the Android project and install it on the target device.

Launch the application.

Point the camera at different objects.

The application should:

1. Capture an image or frame.
2. Resize it.
3. Apply model-compatible preprocessing.
4. Execute MobileNet inference.
5. Identify the highest-scoring class.
6. Display the predicted label.

---

# Step 18: Verify Local Inference

A key objective of the lab is verifying that inference does not depend on a cloud service.

The path should remain:

```text
Image
  ↓
Android Device
  ↓
Local Model
  ↓
Prediction
```

rather than:

```text
Image
  ↓
Internet
  ↓
Remote Server
```

This demonstrates one of the core architectural characteristics of mobile AI.

---

# Step 19: Measure CPU Inference Latency

Measure the time required for inference.

A simple Kotlin measurement pattern is:

```kotlin
val start =
    System.nanoTime()

val probabilities =
    classifier.runInference(
        preprocessedImage
    )

val elapsed =
    System.nanoTime() - start

val latencyMs =
    elapsed / 1_000_000.0
```

Display or log:

```kotlin
println(
    "Inference latency: $latencyMs ms"
)
```

Record several measurements rather than relying on one inference.

---

# Step 20: Establish a CPU Baseline

Record:

| Metric          |       CPU |
| --------------- | --------: |
| Average latency |    Record |
| Minimum latency |    Record |
| Maximum latency |    Record |
| Device          |    Record |
| Model           | MobileNet |
| Input size      |    Record |

Use this baseline before enabling hardware acceleration.

---

# Step 21: Add Optional GPU Acceleration

TensorFlow Lite can use hardware acceleration when supported by the Android device and model.

A simplified GPU delegate configuration is:

```kotlin
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate

val gpuDelegate =
    GpuDelegate()

val options =
    Interpreter.Options().apply {
        addDelegate(
            gpuDelegate
        )
    }

val interpreter =
    Interpreter(
        modelBuffer,
        options
    )
```

The execution architecture becomes:

```text
TensorFlow Lite
      ↓
GPU Delegate
      ↓
Mobile GPU
      ↓
Accelerated Inference
```

---

# Step 22: Manage GPU Delegate Lifecycle

When a delegate is created, it should also be released appropriately when it is no longer needed.

Conceptually:

```text
Application Start
      ↓
Create GPU Delegate
      ↓
Create Interpreter
      ↓
Run Inference
      ↓
Close Interpreter
      ↓
Close Delegate
```

This prevents unnecessary native resource retention.

---

# Step 23: Compare CPU and GPU Execution

Run equivalent inference tests using:

```text
CPU
```

and:

```text
GPU Delegate
```

Record:

| Execution Mode | Average Latency | Notes       |
| -------------- | --------------: | ----------- |
| CPU            |          Record | Baseline    |
| GPU            |          Record | Accelerated |

Calculate speedup:

```text
Speedup =
CPU Latency
───────────
GPU Latency
```

For example:

```text
CPU = 40 ms
GPU = 20 ms

Speedup = 2×
```

Use actual measurements from the device rather than assuming a fixed speedup.

---

# Step 24: Understand Why GPU May Not Always Be Faster

GPU acceleration depends on:

* Mobile GPU architecture
* TensorFlow Lite delegate support
* Model operations
* Tensor size
* Input preprocessing
* Delegate initialization
* Device thermal state
* Workload size

For very small models or low-frequency inference:

```text
GPU Setup / Transfer Overhead
           ↓
May Reduce Expected Benefit
```

Therefore, benchmark the complete application, not just theoretical GPU capability.

---

# Step 25: Validate Classification Quality

Test the application with several objects.

Verify that:

* Camera input is correct.
* Image rotation is handled correctly.
* Images are resized correctly.
* RGB values are processed correctly.
* Normalization matches the model.
* Model output maps correctly to labels.
* Predictions are reasonable.

Incorrect preprocessing is one of the most common causes of poor on-device inference results.

---

# Step 26: Test Offline Operation

Disable network access on the Android device.

Then run the classifier again.

If the application architecture is fully local:

```text
No Internet
    ↓
Model Still Available
    ↓
Inference Still Works
```

This provides a simple demonstration that the application is performing true on-device inference.

---

# Step 27: Understand Mobile AI Advantages

On-device inference can provide several benefits.

## Lower Network Dependency

```text
No Round Trip to Cloud
```

---

## Lower Latency

```text
Input
 ↓
Local Inference
 ↓
Prediction
```

avoids network transmission delay.

---

## Privacy

Sensitive images may remain on the device.

---

## Offline Capability

Inference can continue without Internet connectivity.

---

## Reduced Cloud Inference Cost

Repeated requests do not necessarily consume remote inference infrastructure.

---

# Step 28: Understand Mobile AI Constraints

Mobile inference also has limitations:

* Limited CPU
* Limited GPU
* Limited memory
* Battery consumption
* Thermal throttling
* Model-size constraints
* Device fragmentation
* Hardware compatibility

The central tradeoff is:

```text
Accuracy
   ↕
Model Size
   ↕
Latency
   ↕
Power
   ↕
Memory
```

---

# Step 29: Validate Sustained Inference

Run classification continuously for several minutes.

Observe:

* Inference latency
* Device temperature
* Battery consumption
* UI responsiveness
* Memory behavior
* Camera performance

A short benchmark may not reveal thermal throttling or sustained-load behavior.

---

# Step 30: Improve the Application

Once the basic classifier works, you can extend it with:

* Top-3 or Top-5 predictions
* Confidence scores
* Real-time camera classification
* Image-gallery selection
* Inference latency display
* CPU/GPU switching
* Quantized models
* More efficient preprocessing
* Object detection
* Image segmentation

---

# Step 31: Example Top-3 Predictions

Instead of displaying only one class, you can rank predictions.

Conceptually:

```text
Output Probabilities
       ↓
Sort by Confidence
       ↓
Top 3 Classes
```

For example:

```text
1. Labrador retriever  82%
2. Golden retriever    11%
3. Beagle               3%
```

This can provide more useful feedback when predictions are uncertain.

---

# Step 32: Production Considerations

A production mobile AI application should consider:

## Model Versioning

Track:

```text
Model Name
Model Version
Model Hash
```

---

## Model Updates

Possible approaches include:

```text
Bundled Model
```

or:

```text
Secure Model Download
      ↓
Version Validation
      ↓
Local Storage
```

---

## Privacy

Avoid unnecessary storage or transmission of images.

---

## Performance

Benchmark across multiple representative Android devices rather than only one high-end phone.

---

## Compatibility

Different mobile devices may support different acceleration capabilities.

The application should handle unsupported delegates gracefully.

---

# Step 33: Clean Up

When the activity or classifier is no longer needed, close the TensorFlow Lite Interpreter:

```kotlin
classifier.close()
```

If using a GPU delegate, release it as part of the application lifecycle as well.

This helps prevent native memory and hardware-resource leaks.

---

# Recommended Repository Structure

For your book companion repository:

```text
chapter-22/
├── lab-07-optimize-a-model-with-tensorrt.md
├── lab-11-build-a-mobile-ai-app-with-tensorflow-lite.md
├── lab-11-classifier.kt
└── lab-11-main-activity.kt
```

Model binaries such as:

```text
mobilenet_v2.tflite
```

may be omitted from Git if licensing, repository size, or distribution considerations make it preferable to provide download instructions instead.

---

# Lab Verification Checklist

Before completing the lab, verify that you successfully:

* [ ] Created the Android Studio project.
* [ ] Configured Kotlin.
* [ ] Added TensorFlow Lite dependencies.
* [ ] Created the assets directory.
* [ ] Added the `.tflite` model.
* [ ] Added `labels.txt`.
* [ ] Created the `Classifier` class.
* [ ] Loaded the TensorFlow Lite model.
* [ ] Created the Interpreter.
* [ ] Loaded labels.
* [ ] Connected or reviewed CameraX input.
* [ ] Resized image input.
* [ ] Applied model-compatible preprocessing.
* [ ] Executed local inference.
* [ ] Identified the top output class.
* [ ] Converted the output index to a label.
* [ ] Displayed the prediction.
* [ ] Verified inference on a physical device.
* [ ] Measured CPU latency.
* [ ] Explored GPU acceleration.
* [ ] Compared CPU and GPU performance.
* [ ] Tested classifications with multiple objects.
* [ ] Verified offline inference.
* [ ] Closed inference resources correctly.

---

# Expected Results

At the end of the lab:

* The Android app should contain a packaged TensorFlow Lite model.
* MobileNet should execute entirely on the Android device.
* Camera or image input should be converted into model-compatible tensors.
* The model should return classification probabilities.
* The application should convert the highest-scoring output into a readable label.
* The predicted class should appear in the Android UI.
* CPU inference latency should be measurable.
* GPU acceleration can be evaluated where supported.
* The application should continue performing inference without requiring a cloud model endpoint.

---

# Learning Outcomes

After completing this lab, you should be able to:

* Integrate a TensorFlow Lite model into an Android application.
* Package machine learning models with mobile apps.
* Load TensorFlow Lite models from Android assets.
* Prepare camera input for inference.
* Perform image classification directly on a mobile device.
* Interpret classification probabilities.
* Map output indices to readable labels.
* Explain the architecture of on-device inference.
* Measure mobile AI inference latency.
* Explore GPU delegates.
* Compare CPU and accelerated execution.
* Explain the privacy, latency, and offline advantages of mobile AI.
* Recognize mobile memory, power, thermal, and compatibility constraints.

---

# Key Takeaway

**Mobile AI moves model execution from remote infrastructure directly onto the user's device, allowing applications to perform inference with lower network dependency, improved privacy, and offline capability.**

The complete architecture is:

```text
Camera / Image
      ↓
Preprocessing
      ↓
TensorFlow Lite
      ↓
CPU / GPU Delegate
      ↓
MobileNet
      ↓
Prediction
      ↓
Android UI
```

TensorFlow Lite provides the optimized mobile inference runtime, while MobileNet provides a compact model architecture suitable for resource-constrained devices. The application must still carefully match preprocessing, tensor dimensions, data types, labels, and hardware acceleration settings to the exact model being deployed.

The broader Mobile AI principle is:

```text
Train / Convert Model
        ↓
Package for Mobile
        ↓
Optimize for Device
        ↓
Run Locally
        ↓
Measure on Real Hardware
```

A successful on-device AI application therefore depends not only on the model itself, but also on **correct preprocessing, efficient runtime integration, mobile hardware characteristics, and measured performance on representative devices**.

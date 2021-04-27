package com.example.imagepro;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.SystemClock;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class ObjectDetectorClass {
    int height, width;
    int INPUT_SIZE;
    private Interpreter interpreter;
    List<String> labelList = Arrays.asList(
            "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
            "20", "21", "22", "23", "24", "25", "26", "27", "28"
    );

    ObjectDetectorClass(AssetManager assetManager, String modelPath, int inputSize) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        GpuDelegate gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4);

        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);

        INPUT_SIZE = inputSize;

    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor assetFileDescriptor = assetManager.openFd(modelPath);
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = assetFileDescriptor.getStartOffset();
        long declareLength = assetFileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declareLength);
    }

    public Mat recognizeImage(Mat mat_image){
        // Rotate image 90 degree
//        Mat rotated_mat_image = new Mat();
        Mat rotated_mat_image = mat_image;
//        Core.flip(mat_image.t(), rotated_mat_image, 1);

        // Convert to bitmap
        Bitmap bitmap = null;
        bitmap = Bitmap.createBitmap(rotated_mat_image.cols(), rotated_mat_image.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image, bitmap);

        height = bitmap.getHeight();
        width = bitmap.getWidth();

        // Scale bitmap to input of model
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);


        // Convert bitmap to butebuffer as model input should be in it
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);

        // Define output:
//        float [][][] result = new float[1][10][4];
        Object[] input = new Object[1];
        input[0] = byteBuffer;


        // * Output template
        Map<Integer, Object> output_map = new TreeMap<>();

        float [][][] boxes = new float[1][10][4];
        float[][] scores = new float[1][10];
        float[][] classes = new float[1][10];

        output_map.put(0, boxes);
        output_map.put(1, classes);
        output_map.put(2, scores);

        // * My own output
//        float [][][] output = new float[1][25200][33];

        output_map.put(0, boxes);
        output_map.put(1, classes);
        output_map.put(2, scores);

        Log.wtf("MainActivity", "Run den day");

        long startTime = SystemClock.uptimeMillis();

        interpreter.runForMultipleInputsOutputs(input, output_map);

        Log.wtf("MainActivity", "Run qua day");
        Log.wtf("EstimateTime", Long.toString(SystemClock.uptimeMillis() - startTime));

        // * Postprocess output:
//        postProcess(output[0]);

        Object value = output_map.get(0);
        Object object_classes = output_map.get(1);
        Object object_scores = output_map.get(2);

        for (int i = 0; i < 10; i ++){
            float classValue = (float) Array.get(Array.get(object_classes, 0), i);
            float scoreValue = (float) Array.get(Array.get(object_scores, 0), i);
            if (scoreValue > 0.3){
                Object box1 = Array.get(Array.get(value, 0), i);
                float y1 = (float) Array.get(box1, 0) * height;
                float x1 = (float) Array.get(box1, 1) * width;
                float y2 = (float) Array.get(box1, 2) * height;
                float x2 = (float) Array.get(box1, 3) * width;

                Imgproc.rectangle(rotated_mat_image, new Point(x1,y1), new Point(x2,y2), new Scalar(255,0,0), 2);
//                Imgproc.putText(rotated_mat_image, labelList.get((int)classValue), new Point(x1,y1), 3, 1, new Scalar(100, 100, 100), 2);
            }
        }


        // return back by -90 degree before return
//        Core.flip(rotated_mat_image.t(), mat_image, 0);
        return mat_image;
    }

    private void postProcess(float[][] floats) {

    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap scaledBitmap) {
        ByteBuffer byteBuffer;
        int quant = 1;
        int sizeImage = INPUT_SIZE;
        if (quant == 0) {
            byteBuffer = ByteBuffer.allocateDirect(1*sizeImage*sizeImage*3);
        } else{
            byteBuffer = ByteBuffer.allocateDirect(4*1*sizeImage*sizeImage*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[sizeImage*sizeImage];

        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < sizeImage; i ++)
            for (int j = 0; j < sizeImage; j ++){
                final int val = intValues[pixel++];
                if (quant == 0){
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) (val & 0xFF));
                } else{
                    byteBuffer.putFloat((((val >> 16) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val >> 8) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val) & 0xFF))/255.0f);
                }
            }
        return byteBuffer;
    }
}

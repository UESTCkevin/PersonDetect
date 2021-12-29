import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import com.sun.jna.Native;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.net.MalformedURLException;
import java.nio.file.Paths;

/**
 * @outhor Kevin Pan
 * @date 2021/11/11
 */
public class PersonDetect {
//    static {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//    }

    public static void main(String[] args) throws MalformedURLException {
        LoadDll loadDll = new LoadDll();
        loadDll.loadOpenCV();
        Translator<Image, DetectedObjects> yoloV5Translator = YoloV5Translator.builder().optSynsetArtifactName("person.names").build();
//        Criteria<Image, DetectedObjects> criteria =
//                Criteria.builder()
//                        .setTypes(Image.class, DetectedObjects.class)
//                        .optDevice(Device.cpu())
//                        .optModelUrls(Main.class.getResource("/yolov5s").getPath())
//                        .optModelName("person.torchscript.pt")
//                        .optTranslator(yoloV5Translator)
//                        .optEngine("PyTorch")
//                        .build();
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class)
                        .optDevice(Device.cpu())
                        //.optModelUrls(Main.class.getResource("/yolov5s").getPath())
                        //.optModelName("person.onnx")
                        .optModelPath(Paths.get("src/main/resources/yolov5s/person.onnx"))
                        .optTranslator(yoloV5Translator)
                        .optEngine("OnnxRuntime")
                        .build();
        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            FFmpegFrameGrabber fFmpegFrameGrabber = FFmpegFrameGrabber.createDefault("src/main/resources/video/person.mp4");
            fFmpegFrameGrabber.start();
            int videoLength = fFmpegFrameGrabber.getLengthInFrames();
            Frame f;
            int i = 0;
            while (i < videoLength) {
                f = fFmpegFrameGrabber.grabImage();
                Java2DFrameConverter converter = new Java2DFrameConverter();
                BufferedImage bi = converter.getBufferedImage(f);
                Mat oldFrame = BufImg2Mat(bi, BufferedImage.TYPE_3BYTE_BGR, CvType.CV_8UC3);
                Mat frame = new Mat();
                Imgproc.resize(oldFrame, frame, new Size(640, 640));
                detect(frame, model);
                HighGui.imshow("yolov5", frame);
                HighGui.waitKey(10);
                i++;
            }
            fFmpegFrameGrabber.stop();
//            Mat oldFrame = Imgcodecs.imread("E:\\Fire\\R-C.jpg");
//            Mat frame = new Mat();
//            Imgproc.resize(oldFrame, frame, new Size(640, 640));
//            detect(frame, model);
//            HighGui.imshow("yolov5", frame);
//            HighGui.waitKey(10);
        } catch (RuntimeException | ModelException | TranslateException | IOException e) {
            e.printStackTrace();
        }
    }

    private static Rect rect = new Rect();
    private static Scalar color = new Scalar(0, 255, 0);

    public static void detect(Mat frame, ZooModel<Image, DetectedObjects> model) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        BufferedImage bufferedImage = Mat2BufImg(frame, ".jpg");
        Image img = ImageFactory.getInstance().fromImage(bufferedImage);
        long startTime = System.currentTimeMillis();
        try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            long start = System.currentTimeMillis();
            DetectedObjects results = predictor.predict(img);
            long end = System.currentTimeMillis();
            System.out.println("共耗时"+(end-start)+"毫秒");
            //System.out.println(results);
            for (DetectedObjects.DetectedObject obj : results.<DetectedObjects.DetectedObject>items()) {
                BoundingBox bbox = obj.getBoundingBox();
                Rectangle rectangle = bbox.getBounds();
                String showText = String.format("%s: %.2f", obj.getClassName(), obj.getProbability());
                rect.x = (int) rectangle.getX();
                rect.y = (int) rectangle.getY();
                rect.width = (int) rectangle.getWidth();
                rect.height = (int) rectangle.getHeight();
                // 画框
                Imgproc.rectangle(frame, rect, color, 1);
                //画名字
                Imgproc.putText(frame, showText, new Point(rect.x, rect.y), Imgproc.FONT_HERSHEY_COMPLEX, rectangle.getWidth() / 100, color);
            }
        }
        //System.out.println(String.format("%.2f", 1000.0 / (System.currentTimeMillis() - startTime)));
    }
    /*
        将Mat类型的图片转化为BufferedImage类型
     */
    private static BufferedImage Mat2BufImg (Mat matrix, String fileExtension) {
        MatOfByte mob = new MatOfByte();
        Imgcodecs.imencode(fileExtension, matrix, mob);
        byte[] byteArray = mob.toArray();
        BufferedImage bufImage = null;
        try {
            InputStream in = new ByteArrayInputStream(byteArray);
            bufImage = ImageIO.read(in);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return bufImage;
    }

    /*
        将BufferedImage类型的图片转化为Mat类型
     */
    private static Mat BufImg2Mat (BufferedImage original, int imgType, int matType) {
        if (original.getType() != imgType) {
            BufferedImage image = new BufferedImage(original.getWidth(), original.getHeight(), imgType);
            Graphics2D g = image.createGraphics();
            try {
                g.setComposite(AlphaComposite.Src);
                g.drawImage(original, 0, 0, null);
            } finally {
                g.dispose();
            }
        }
        byte[] pixels = ((DataBufferByte) original.getRaster().getDataBuffer()).getData();
        Mat mat = Mat.eye(original.getHeight(), original.getWidth(), matType);
        mat.put(0, 0, pixels);
        return mat;
    }
    private static class LoadDll{
        public void loadOpenCV() {
            try {
                InputStream inputStream = null;
                File fileOut = null;
                String osName = System.getProperty("os.name");
                System.out.println(osName);

                if (osName.startsWith("Windows")) {
                    int bitness = Integer.parseInt(System.getProperty("sun.arch.data.model"));
                    if (bitness == 32) {
                        System.out.println(32 + " bit load success!");
                        inputStream = this.getClass().getResourceAsStream("/opencv/windows/x86/opencv_java452.dll");
                        fileOut = File.createTempFile("lib", ".dll");
                    } else if (bitness == 64) {
                        System.out.println(64 + " bit load success!");
                        inputStream = this.getClass().getResourceAsStream("/opencv/opencv_java452.dll");
                        if (inputStream == null)
                            System.out.println("fail");
                        fileOut = File.createTempFile("lib", ".dll");
                    } else {
                        inputStream = this.getClass().getResourceAsStream("/opencv/windows/x86/opencv_java300.dll");
                        fileOut = File.createTempFile("lib", ".dll");
                    }
                } else if (osName.equals("Mac OS X")) {
                    System.out.println("Mac OS load success!");
                    inputStream = this.getClass().getResourceAsStream("/opencv/mac/libopencv_java300.dylib");
                    fileOut = File.createTempFile("lib", ".dylib");
                }else {
                    System.out.println("Linux load success!");
                    inputStream = this.getClass().getResourceAsStream("/opencv/bopencv_java452.so");
                    fileOut = File.createTempFile("lib", ".so");
                }
                if (fileOut != null) {
                    OutputStream outputStream = new FileOutputStream(fileOut);
                    byte[] buffer = new byte[1024];
                    int length;

                    while ((length = inputStream.read(buffer)) > 0) {
                        outputStream.write(buffer, 0, length);
                    }

                    inputStream.close();
                    outputStream.close();
                    System.load(fileOut.toString());
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}

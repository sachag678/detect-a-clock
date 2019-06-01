package main.objectdetection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractor;

public class ObjectDetection {
	
	public static void detectCircleUsingHoughCircleTransform(String filename) {
		Mat src = Imgcodecs.imread(filename, Imgcodecs.IMREAD_COLOR);
		
		//convert to gray
        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
       
        //blur
        Imgproc.medianBlur(gray, gray, 5); 
         
        //detect circles
        Mat circles = new Mat();
        Imgproc.HoughCircles(gray, circles, Imgproc.HOUGH_GRADIENT, 1.0,
                (double)gray.rows(), // change this value to detect circles with different distances to each other
                150.0, 30.0, 200, 0); // change the last two parameters
                // (min_radius & max_radius) to detect larger circles 
        
        for (int x = 0; x < circles.cols(); x++) {
            double[] c = circles.get(0, x);
            Point center = new Point(Math.round(c[0]), Math.round(c[1]));
            // circle center
            Imgproc.circle(src, center, 1, new Scalar(0,100,100), 3, 8, 0 );
            // circle outline
            int radius = (int) Math.round(c[2]);
            Imgproc.circle(src, center, radius, new Scalar(255,0,255), 3, 8, 0 );
        }
        
        //draw circles onto the img.
        HighGui.imshow("detected circles", src);
        HighGui.waitKey();
		
	}
	
	public static boolean detectCircleUsingContours(String filename) {
		System.out.println(filename);	
		Mat src = Imgcodecs.imread(filename, Imgcodecs.IMREAD_GRAYSCALE);
		
		Mat result = new Mat();
		Imgproc.adaptiveThreshold(src, result, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 3, 0);
		
		//HighGui.imshow("threshold", result);
		//HighGui.waitKey();
		
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        
       // System.out.println("Done converting image..");

        Imgproc.findContours(result, contours, new Mat(), Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);
        
        Mat contourImg = new Mat(result.size(), result.type());
      //  System.out.println(contours.size());
        MatOfPoint points = contours.get(0);
        
        //then use center point to calculate the radius of the results and see if it varies
        double xMax = 0;
        double xMin = 0;
        double yMax = 0;
        double yMin = 0;
       
        
        for(int i=0;i<points.size(0);i++) {
				double [] point = (points.get(i, 0));
				if(i==0) {
					xMin = point[0];
					yMin = point[1];
				}
				if(point[0]>xMax) xMax = point[0];
				if(point[0]<xMin) xMin = point[0];
				if(point[1]>yMax) yMax = point[1];
				if(point[1]<yMin) yMin = point[1];
				
				//System.out.println(Arrays.toString(point));
        }
        
        double [] center_point = new double [] {(xMax-xMin)/2, (yMax-yMin)/2};
       // System.out.println(xMax +" "+ xMin+ " "+ yMax+ " "+ yMin);
        //System.out.println(Arrays.toString(center_point));
        
        double maxR = 0;
        double minR = 0;
        for(int i=0;i<points.size(0);i++) {
			double [] point = (points.get(i, 0));
        	double radius = Math.sqrt(Math.pow(point[0]-center_point[0], 2) + Math.pow(point[1]-center_point[1], 2));
			if(i==0) {minR = radius;}
			if(maxR<radius) {maxR = radius;}
			if(minR>radius) {minR = radius;}
        }
        //System.out.println(maxR +" " +minR);
        	
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(contourImg, contours, i, new Scalar(255, 255, 255), 1);
        }
        
       // HighGui.imshow("countour", contourImg);
        //HighGui.waitKey();
        
        int tol = 39;
        
        if(maxR<center_point[0]+center_point[0]*tol/100 && minR>center_point[0]-center_point[0]*tol/100) {
        	return true;
        }else {
        	return false;
        }
	}
	
	public static String predict(Mat result) {
		return "1";
	}
	
	public static boolean detectNumber(String filename, String filename2, int expected) {
		
		Mat src1 = Imgcodecs.imread(filename, Imgcodecs.IMREAD_GRAYSCALE);
		Mat src2 = Imgcodecs.imread(filename2, Imgcodecs.IMREAD_GRAYSCALE);
		
		Mat dst = new Mat();
		Core.absdiff(src1, src2, dst);
		
		Mat result = new Mat();
		Imgproc.adaptiveThreshold(dst, result, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 3, 0);
		
		//HighGui.imshow("threshold", result);
		//HighGui.waitKey();
		MatOfPoint2f approxCurve = new MatOfPoint2f();
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        
       // System.out.println("Done converting image..");

        Imgproc.findContours(result, contours, new Mat(), Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);
        
        List<Mat> digits = new ArrayList<>();
        
        for (int i = 0; i < contours.size(); i++) {

            //Convert contours(i) from MatOfPoint to MatOfPoint2f
            MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(i).toArray() );

            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f, true)*0.02;
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

            //Convert back to MatOfPoint
            MatOfPoint points = new MatOfPoint( approxCurve.toArray() );

            // Get bounding rect of contour
            org.opencv.core.Rect rect = Imgproc.boundingRect(points);
            
            digits.add(new Mat(result, rect));
            
            Imgproc.rectangle(dst, rect.tl(), rect.br(), new Scalar(255, 255, 255), 1);
        }
        String res = "";
        for(int i=0;i<digits.size();i++) {
			res+=ObjectDetection.predict(digits.get(i));
        }
        return res.equals(String.valueOf(expected));
        
        //convert items in the digits into bitmaps to be passed into the network
        
       //HighGui.imshow("hi", dst);
        //HighGui.waitKey();
		
	}
	
	public static void main(String [] args) {
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        assert ObjectDetection.detectCircleUsingContours("img/clock1.png");
        assert !ObjectDetection.detectCircleUsingContours("img/clock2.png");
        assert !ObjectDetection.detectCircleUsingContours("img/Bumpy.png");
        assert !ObjectDetection.detectCircleUsingContours("img/AngledEllipse.png");
        assert ObjectDetection.detectCircleUsingContours("img/GoodCircle1.png");
        assert ObjectDetection.detectCircleUsingContours("img/GoodCircle2.png");
        assert !ObjectDetection.detectCircleUsingContours("img/Hat.png");
        assert !ObjectDetection.detectCircleUsingContours("img/Square.png");
        assert !ObjectDetection.detectCircleUsingContours("img/Triangle.png");
        assert !ObjectDetection.detectCircleUsingContours("img/TallEllipse.png");
        assert ObjectDetection.detectCircleUsingContours("img/GoodClock/circle-GoodClock.png");
        
        assert detectNumber("img/GoodClock/circle-GoodClock.png", "img/GoodClock/GoodClock-0B.png", 111);
	}
}

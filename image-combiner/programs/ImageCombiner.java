package Programs;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

/**
* Combines similarly named images into a single image.
*    Loads images from 'src/images' and saves the combined images to 'src/output'
*/
public class ImageCombiner {
   
   public static void main(String args[]) {
      // Maps the original image name to a list of all relevant image filepaths
      HashMap<String, List<String>> filePathMap = new HashMap<String, List<String>>();
      File[] files = new File[3];
      files[0] = new File("src/images/emotions-sorted");
      files[1] = new File("src/images/saliency-maps");
      files[2] = new File("src/images/output_per_class");
      fillFilePathMap(filePathMap, files);
      /*
      *   Debugging statement for processImages()
      *      Processes a single group of images
      *
      String originalName = "human_sad_10.png";
      processImages(filePathMap.get(originalName), originalName, new File("src/output/"+originalName+"_combined.png"));
      */
      
      // Goes through the entire filePathMap in order to combine images
      for (String originalName : filePathMap.keySet()) {
         List<String> filePaths = filePathMap.get(originalName);
         if (filePaths.size() > 1) {
            processImages(filePathMap.get(originalName), originalName, new File("src/output/"+originalName+"_combined.png"));
         }
      }
   }
   
   /**
   * Sorts through all files in the given directory by adding them to the filePathMap based on their
   *   original file name
   */
   public static void fillFilePathMap (Map<String, List<String>> map, File[] files) {
      for (File file : files) {
         if (file != null) {
            // Processes File based on its original file name
            if (!file.isDirectory()) {
               String fileName = file.getName();
               int index = fileName.indexOf(".png");
               if (index != 1) {
                  String originalName = fileName.substring(0, index + 4);
                  if (!map.containsKey(originalName)) {
                     map.put(originalName, new ArrayList<String>());
                  }
                  map.get(originalName).add(file.getPath());
               }
               // Processes sub-directories
            } else {
               fillFilePathMap(map, file.listFiles());
            }
         }
      }
   }
   
   /**
   * Combines the images represented by each String in filePaths into a single image.
   *   The new image is saved to the given output file.
   */
   public static void processImages (List<String> filePaths, String originalName, File output) {
      List<BufferedImage> images = new ArrayList<BufferedImage>();
      List<File> files = new ArrayList<File>();
      // Creates all necessary images
      for (int i = 0; i < filePaths.size(); i++) {
         files.add(new File(filePaths.get(i)));
         try {
            images.add(ImageIO.read(files.get(i)));
         } catch (IOException e) {
            e.printStackTrace();
         }
      }
      // Defines characteristics of individual images
      int standardWidth = images.get(0).getWidth();
      int standardHeight = images.get(0).getHeight();
      // Defines characteristics of the output image
      int imgPerLine = 7;
      int imgHeight = standardHeight * (((images.size()-1)/imgPerLine)+1);
      int imgWidth = standardWidth * Math.min(images.size(), imgPerLine);
      BufferedImage newImage = new BufferedImage(imgWidth, imgHeight, BufferedImage.TYPE_INT_ARGB);
      Graphics2D g2 = (Graphics2D)newImage.getGraphics();
      // Draws each image on newImage and labels it accordingly
      for (int i = 0; i < images.size(); i++) {
         // Draws the image, starting a new row once a row contains imgPerLine images
         g2.drawImage(images.get(i), standardWidth*(i%imgPerLine), standardHeight*(i/imgPerLine), null);
         String name = "";
         // Special naming convention for the first entry, the original photo
         if (i != 0) {
            name = files.get(i).getName().substring(originalName.length()+1);
         } else {
            name = originalName;
         }
         // Draws label beneath the drawn image
         g2.drawString(name, standardWidth*(i%imgPerLine), standardHeight*((i/imgPerLine)+1) - 20);
      }
      // Saves the new image as a png to the output file
      try {
         ImageIO.write(newImage, "png", output);
      } catch (IOException e) {
         e.printStackTrace();
      }
   }
   
}

#include "utils.h"

int main(int argc,char** argv)
{
    Mat img,segmented;
    String basePath = "G:/tcc_imagens/";
    String arffClass = "integral";
    String relation = "bolachas";
    String classes = "{recheio,recheioDef,normal,normalDef,integral,integralDef}";
    const char* archive = "G:/features.arff";
    int i=1;
    int limit = 29;
    int total = 0;

    cout << "Generating arff header..." << endl;
    if(!generateArffHeader(archive,relation,classes)){
        cout << "Err generating " << archive << "header" << endl;
        return 0;
    }
    cout << "Ready!" << endl;
    cout << "----------------------------------------------------------------------------------------------------" << endl;

    cout << "Processing images..." << endl;
    for(i=1; i<=limit;i++){
        //Builds the image url.
        String typePath = arffClass + "/";
        String path = basePath + typePath + "1 ("+toString(i)+").jpg";
        //Reads the image.
        img = imread(path,IMREAD_COLOR);
        //Preprocessing and segmentation.
        segmented = preProcessing(img);
        //Feature extraction.
        shapeFeature(segmented);//Hu Moments
        textureFeature(segmented);//Haralick descriiptors
        colorFeature(segmented);//GHC
        //Arff generation.
        if(!generateArffData(archive,arffClass)){
            cout << "Err generating " << archive << " data" << endl;
            break;
        }
        cout << path + "--> ok" << endl;
        total++;
        //Gets the next class and quantity of images when the index represents the last image of the current class.
        if(i == limit){
            getNextClass(arffClass,limit);
            i = 0;
        }
    }
    cout << "----------------------------------------------------------------------------------------------------" << endl;
    cout << "All done!" << endl;
    cout << "Total images: " << total << endl;
    cout << archive << " generated successfully!" << endl;
    return 0;
}

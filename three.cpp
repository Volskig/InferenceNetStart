#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace InferenceEngine;

Mat frame, blob;
float confThreshold = 0.7;

void postprocess(Mat& frame, const Mat& outs)
{
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]   
   
    float* data = (float*)outs.data;
    for (size_t i = 0; i < outs.total(); i += 7)
    {
        float confidence = data[i + 2];			
        if (confidence > confThreshold)
        {		
            int left   = (int)(data[i + 3] * frame.cols);
            int top    = (int)(data[i + 4] * frame.rows);
            int right  = (int)(data[i + 5] * frame.cols);
            int bottom = (int)(data[i + 6] * frame.rows);			
			rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));              
        }
    }
}

static inline void genData(const std::vector<size_t>& dims, Mat& m, Blob::Ptr& dataPtr)
{
    //m.create(std::vector<int>(dims.begin(), dims.end()), CV_32F);
	blobFromImage(frame, m, 1, Size(300, 300));
    dataPtr = make_shared_blob<float>({Precision::FP32, dims, Layout::ANY}, (float*)m.data);
}

static inline void getData(const std::vector<size_t>& dims, Mat& m, Blob::Ptr& dataPtr)
{		
	blobFromImage(frame, blob, 1, Size(300, 300));	
    dataPtr = make_shared_blob<float>({Precision::FP32, dims, Layout::ANY}, (float*)blob.data);
}

void runIE(const std::string& xmlPath, const std::string& binPath, std::map<std::string, cv::Mat>& inputsMap, std::map<std::string, cv::Mat>& outputsMap)
{	
    CNNNetReader reader;
    reader.ReadNetwork(xmlPath);
    reader.ReadWeights(binPath);

    CNNNetwork net = reader.getNetwork();

    InferenceEnginePluginPtr enginePtr;
    InferencePlugin plugin;

    ExecutableNetwork netExec;
    InferRequest infRequest;

    try
    {
        auto dispatcher = InferenceEngine::PluginDispatcher({""});
        enginePtr = dispatcher.getPluginByDevice("CPU");

	    IExtensionPtr extension = make_so_pointer<IExtension>("libcpu_extension.so");
        enginePtr->AddExtension(extension, 0);
        plugin = InferencePlugin(enginePtr);
        netExec = plugin.LoadNetwork(net, {});

        infRequest = netExec.CreateInferRequest();
    }
    catch (const std::exception& ex)
    {
        CV_Error(Error::StsAssert, format("Failed to initialize Inference Engine backend: %s", ex.what()));
    }		

    // Fill input blobs.
    inputsMap.clear();
    BlobMap inputBlobs;	
    for (auto& it : net.getInputsInfo())   
	{			
        genData(it.second->getTensorDesc().getDims(), inputsMap[it.first], inputBlobs[it.first]);
    }
    infRequest.SetInput(inputBlobs);	

    // Fill output blobs.
    outputsMap.clear();
    BlobMap outputBlobs;
    for (auto& it : net.getOutputsInfo())
    {
        genData(it.second->getTensorDesc().getDims(), outputsMap[it.first], outputBlobs[it.first]);
    }	
    infRequest.SetOutput(outputBlobs);
	
    infRequest.Infer();
}

int main(int argc, char* argv[])
{
	VideoCapture cap(0);	
	cap.read(frame);
  	//====================================================================================================================================
    std::string xmlPath = "/home/volskig/src/weights/face-detection-retail-0004/FP32/face-detection-retail-0004.xml";
    std::string binPath = "/home/volskig/src/weights/face-detection-retail-0004/FP32/face-detection-retail-0004.bin";

    std::map<std::string, cv::Mat> inputsMap, ieOutputsMap; 

    runIE(xmlPath, binPath, inputsMap, ieOutputsMap); 

	postprocess(frame, ieOutputsMap["detection_out"]);
	//====================================================================================================================================
	while(true)
    {		
		imshow("window", frame);
		if ((int)waitKey(10) == 27)
		{
			break;
		}
	}    
    return 0;
}


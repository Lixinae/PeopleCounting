import numpy as np
import Voxel

class VxlCameraInfo:
		
		deviceResolution = {
			'OPT8320' : (60,80),
			'OPT8241' : (240,320)
		}
		
		def __init__(self, device):
			self.system = Voxel.CameraSystem()
			self.resolution = VxlCameraInfo.deviceResolution[device]

class VxlFrame:

	def __init__(self, attributes):
		self.amplitude = attributes["amplitude"]
		self.depth = attributes["depth"]
		self.shape = self.amplitude.shape

class VxlVideo:

	def __init__(self, frames):
		self.frames = frames

	def __iter__(self):
		return iter(self.frames)

	def __getitem__(self, index):
		return self.frames[index]
		
	def __len__(self):
		return len(self.frames)

	@staticmethod
	def read(vxlFile, cameraInfo):
		reader = Voxel.FrameStreamReader(vxlFile, cameraInfo.system)
		if not reader.isStreamGood():
			raise ValueError("Stream is not good: " + vxlFile)
		frames = list()
		for i in range(reader.size()):
			if not reader.readNext():
				raise ValueError("Failed to read frame number " +  str(i))
			rawProcessedFrame = Voxel.ToF1608Frame.typeCast(reader.frames[Voxel.DepthCamera.FRAME_RAW_FRAME_PROCESSED])
			depthFrame = Voxel.DepthFrame.typeCast(reader.frames[Voxel.DepthCamera.FRAME_DEPTH_FRAME])
			attributes = {
				"ambient" : np.array(rawProcessedFrame._ambient, copy=True).reshape(cameraInfo.resolution),
				"amplitude" : np.array(rawProcessedFrame._amplitude, copy=True).reshape(cameraInfo.resolution),
				"phase" : np.array(rawProcessedFrame._phase, copy=True).reshape(cameraInfo.resolution),
				"depth" : np.array(depthFrame.depth, copy=True).reshape(cameraInfo.resolution)
			}
			frames.append(VxlFrame(attributes))
		reader.close()
		return VxlVideo(frames)
		
	@staticmethod
	def readAsAvgImage(vxlFile, cameraInfo):
		video = VxlVideo.read(vxlFile, cameraInfo)
		depth = np.zeros(cameraInfo.resolution)
		amplitude = np.zeros(cameraInfo.resolution)
		for frame in video:
			amplitude += frame.amplitude
			depth += frame.depth
		amplitude = amplitude/len(video)
		depth = depth/len(video)
		return VxlFrame({
			"amplitude" : amplitude,
			"depth" : depth
		})

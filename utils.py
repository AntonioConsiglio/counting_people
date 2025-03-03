import time

class PassCount:
	def __init__(self,centroid,bin,bout):
		self.linein = None
		self.lineout = None
		self.middle = None
		self.centroid = centroid
		self.counted = False
		self.c_max = centroid
		self.c_min = centroid
		self.step = 0
		self.bin = bin
		self.bout = bout
	
	def update_centroid(self,centroid):
		self.centroid = centroid
		if centroid > self.c_max:
			self.c_max = centroid
		elif centroid < self.c_min:
			self.c_min = centroid
		self.step+=1
	
	def count(self):
		value = 0
		if self.counted or self.step == 0:
			return value
		
		if self.linein is not None and self.lineout is not None:
			if self.linein > self.lineout:
				value = 1
				self.counted = True
			else:
				value = -1
				self.counted = True
		elif self.middle is not None:
			
			if self.linein is not None:
				if self.linein > self.middle:
					if self.bout < self.c_min < self.bin:
						print("sono qui")
						value = 1
						self.counted = True
				# else:
				# 	value = -1
				# 	self.counted = True
			elif self.lineout is not None:
				if self.lineout > self.middle:
					if self.bout < self.c_max < self.bin:
						print("sono qui")
						value = -1
						self.counted = True
				# else:
					
				# 		value = -1
				# 		self.counted = True
				

		return value

	def _calculate_centroid(self,xy):
			return int((xy[-1]-xy[1])//2 + xy[1])
		
class LineCounter:
	def __init__(self,p1=None,p2=None,gap=20):
		assert p1 is not None 
		if p2 is None:
			self.x = None
			self.y = p1[1]
			self.y_out = self.y-gap
			self.line = True
		else:
			self.x = [p1[0],p2[0]]
			self.y = [p1[1],p2[1]]
			self.line = False
		self.counter_in = 0
		self.counter_out = 0
		self.IDs = {}
	
	def _calculate_centroid(self,xy):
		if self.line:
			return int((xy[-1]-xy[1])//2 + xy[1])
	
	def _calculate_direction(self,arr):
		ycord = arr[:,1]
		differences = [ycord[n] - j for (n,j) in enumerate(ycord[1:])]
		direction = sum(differences)
		if direction > 10:
			direction = False
		elif direction < -10:
			direction =  True
		else:
			direction = None
		return direction

	def check_pass(self,detections,annotator):
		for detection in detections:
			centroid = self._calculate_centroid(detection[0])
			bbid = detection[4]
			if bbid not in self.IDs.keys():
				self.IDs[bbid] = PassCount(centroid,self.y,self.y_out)
			else:
				self.IDs[bbid].update_centroid(centroid)
			tocheck = self.IDs[bbid]
			#direction = self._calculate_direction(annotator.trace.get(bbid))
			if self.line:
				if tocheck.centroid > self.y:
					if tocheck.linein is None:
						tocheck.linein = time.time()
					# self.IDs.append(bbid)
					# self.counter +=1
				elif tocheck.centroid < self.y_out : #and bbid not in self.IDs:
					if tocheck.lineout is None:
						tocheck.lineout = time.time()
						# self.IDs.append(bbid)
						# self.counter -=1
				else:
					if tocheck.middle is None:
						tocheck.middle = time.time()
			count = tocheck.count()
			if count > 0:
				self.counter_in += 1
			elif count < 0:
				self.counter_out += 1
			
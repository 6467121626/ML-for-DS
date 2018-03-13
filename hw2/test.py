import time
now =  time.time()
realtime =  time.strftime("%H%M ", time.localtime(now))
minutes = int(realtime[:2])*60 + int(realtime[2:])
print now
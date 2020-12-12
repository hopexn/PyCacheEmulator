import collections


class ARC:
    def __init__(self, size):
        self.cache = set()
        self.c = size
        self.p = 0
        self.T1 = collections.deque()
        self.B1 = collections.deque()
        self.T2 = collections.deque()
        self.B2 = collections.deque()
        self.hit_cnt = 0
        self.req_cnt = 0
    
    def replace(self, item):
        if len(self.T1) >= 1 and ((item in self.B2 and len(self.T1) == self.p) or len(self.T1) > self.p):
            old = self.T1.pop()
            self.B1.appendleft(old)
        else:
            old = self.T2.pop()
            self.B2.appendleft(old)
        
        self.cache.remove(old)
    
    def get_hit_ratio(self):
        return self.hit_cnt / (self.req_cnt + 1e-6)
    
    def re(self, item):
        self.req_cnt += 1
        # Case I
        if (item in self.T1) or (item in self.T2):
            self.hit_cnt += 1
            if item in self.T1:
                self.T1.remove(item)
            
            elif item in self.T2:
                self.T2.remove(item)
            
            self.T2.appendleft(item)
        # Case II
        elif item in self.B1:
            self.p = min(self.c, self.p + max(len(self.B2) / len(self.B1) * 1., 1))
            self.replace(item)
            self.B1.remove(item)
            self.T2.appendleft(item)
            self.cache.add(item)
        # Case III
        elif item in self.B2:
            self.p = max(0, self.p - max(len(self.B1) / len(self.B2) * 1., 1))
            self.replace(item)
            self.B2.remove(item)
            self.T2.appendleft(item)
            self.cache.add(item)
        # Case IV (Inserting a new item)
        elif (item not in self.T1) or (item not in self.B1) or (item not in self.T2) or (item not in self.B2):
            # Case (i)
            if len(self.T1) + len(self.B1) == self.c:
                if len(self.T1) < self.c:
                    self.B1.pop()
                    self.replace(item)
                else:
                    old = self.T1.pop()
                    self.cache.remove(old)
                    # Case (ii)
            elif len(self.T1) + len(self.B1) < self.c <= (
                    len(self.T1) + len(self.B1) + len(self.T2) + len(self.B2)):
                if (len(self.T1) + len(self.B1) + len(self.T2) + len(self.B2)) == 2 * self.c:
                    self.B2.pop()
                self.replace(item)
            
            self.T1.appendleft(item)
            self.cache.add(item)
        else:
            print("There is an error.")

import numpy as np
import tensorflow as tf
import scipy.sparse.linalg

def makeLS(p):
    if p[0] == 'dense':
        def LS(A,b):
            n = tf.reduce_prod(b.shape)
            br = tf.reshape(b,(n,))
            Ar = tf.reshape(A,(n,n)) + tf.eye(n,dtype=tf.float64)*p[1]
            xr = tf.linalg.solve(Ar,tf.expand_dims(br,1))
            x  = tf.reshape(xr,b.shape)
            return x

    elif p[0] == 'cg':
        def LS(A,b):
            n = tf.reduce_prod(b.shape)
            br = tf.reshape(b,(n,)).numpy()
            Ar = tf.reshape(A,(n,n)).numpy()
            sol = scipy.sparse.linalg.cg(Ar,br,tol=p[1],maxiter=p[2])
            xr = sol[0]
            x = tf.reshape(xr,b.shape)
            return x
    return LS



class model:
    def __init__(self,NN,nvars,gdvars,opt,LS,alpha=1e-4,rho=0.7):
        self.NN = NN
        
        self.nvars = nvars
        self.gdvars = gdvars
        
        self.loss = tf.keras.losses.sparse_categorical_crossentropy
        self.opt=opt

        self.LS = LS

        self.alpha = alpha
        self.rho = rho


    @tf.function
    def getJ(self,x,ytrue):
        J = tf.reduce_mean(self.loss(ytrue,self.NN(x)))
        return J
    
    @tf.function
    def getdJ(self,x,ytrue):
        with tf.GradientTape() as gg:
            gg.watch(self.nvars)
            with tf.GradientTape() as g:
                g.watch(self.nvars)
                J = self.getJ(x,ytrue)
            dJ = g.gradient(J,[self.nvars])[0]
        d2J = gg.jacobian(dJ,[self.nvars])[0]
        return J,dJ,d2J

    def newton(self,x,ytrue):
        J,dJ,d2J = self.getdJ(x,ytrue)
        nvars_old = tf.Variable(self.nvars)
        dnvars = self.LS(d2J,dJ)
        lamb = 1.
        self.nvars.assign(nvars_old-lamb*dnvars)
        J_new = self.getJ(x,ytrue)
        while J_new > J - self.alpha*lamb*tf.reduce_sum(dJ*dnvars):
            lamb = lamb*self.rho
            self.nvars.assign(nvars_old-lamb*dnvars)
            J_new = self.getJ(x,ytrue)

    def newton_conv(self,x,ytrue,tol=1e-6,max_iter=100000):
        iter = 0
        J1 = self.getJ(x,ytrue)
        J2 = np.inf
        while abs(J2 - J1) > tol:
            iter+=1
            self.newton(x,ytrue)
            J2 = J1
            J1 = self.getJ(x,ytrue)

            if iter>max_iter:
                break
        return iter,(J1-J2).numpy()
    
    
    @tf.function
    def gd(self,x,ytrue):
        with tf.GradientTape() as g:
            g.watch(self.gdvars)
            J =  self.getJ(x,ytrue)
        dJ = g.gradient(J,self.gdvars)
        self.opt.apply_gradients(zip(dJ,self.gdvars))

    
    def getacc(self,x,ytrue):
        return tf.keras.metrics.Accuracy(dtype=tf.float64)(ytrue,tf.argmax(self.NN(x),1)).numpy()





# Skript for generating the Functions as a String
# Idea is, these Classes should create and return a string which in the end via excec(....., globals()) can be turned
# into a python function and in the end into a model for fitting using lmfit
import math as m
from asteval import asteval

class Gen_Lorentz():
    def __init__(self,fit_num):
        self.num = fit_num

    def get_str(self):
        str_func_args = str() # Set empty str
        str_body = str() # Set empty str
        str_end = "slope*B+offset" # Set linear fkt
        for i in range(1,self.num+1):   # create function/s
            str_func_args += ", dB{0}, R{0}, A{0}".format(i)
            str_body += "-32*A{0}*dB{0}**3*(B-R{0})/(3*dB{0}**2+4*(B-R{0})**2)**2".format(i)
        func = "def model_fit_func(B,slope,offset{0}):\n\treturn({1} + {2})".format(str_func_args,str_body,str_end)
        return func

class Gen_Dyson():
    def __init__(self,fit_num):
        self.num = fit_num

    def get_str(self):
        str_func_args = str()
        str_body = str()
        str_end = "slope*B+offset"
        for i in range(1,self.num+1):
            str_func_args += ", alpha{0}, dB{0}, R{0}, A{0}".format(i)
            if i == 1:
                str_body += "(4*A{0}*dB{0}**2*(3*alpha{0}*dB{0}-4*alpha{0}*(B-R{0})**2-8*m.sqrt(3)*dB{0}*(B-R{0})))/(m.sqrt(3)*(4*(B-R{0})**2+3*dB{0}**2)**2)".format(i)
            else:
                str_body += "+(4*A{0}*dB{0}**2*(3*alpha{0}*dB{0}-4*alpha{0}*(B-R{0})**2-8*m.sqrt(3)*dB{0}*(B-R{0})))/(m.sqrt(3)*(4*(B-R{0})**2+3*dB{0}**2)**2)".format(
                    i)
        func = "def model_fit_func(B, slope,offset{0}):\n\treturn({1} + {2})".format(str_func_args,str_body,str_end)
        return func

#-------Example on how to get the function string and excecute it to a pyhon function
#obj = Gen_Dyson(10)    # Save Object with 10 Functions to obj
#model = obj.get_str()  # call gen_str() to generate str
#exec(model,globals())   # excecute to python function in global namespace

class Fit_Models():
    # Todo: Make MODELS as Object; Args, Bounds, Stepsize... as property
    def __init__(self):
        self.load_models()

    def load_models(self):
        # Loads a .txt file where the models are stored
        MODELS = dict()

        with open("lib/Models.txt", 'r') as file:
            model = dict()
            #i = 0
            for line in file:
                if line[0] == "#": # Name
                    name = line[1:-1]
                    model["name"] = name
                    #print(name)
                elif line[:4] == "Args": # Arguments
                    args = line[6:-1]
                    args_fmt = args.replace(",", r"{0}") + r"{0}"
                    args_fmt = ", " + args_fmt.replace(" ", ", ")
                    model["args"] = args.split(",")
                    model["args_fmt"] = args_fmt
                    #print(args)
                elif line[:6] == "Bounds":
                    bounds_raw = line[8:-1].split(", ")
                    bounds = []
                    for tpl in bounds_raw:
                        big = "9999999999"
                        tpl = eval(tpl.replace('Nan', big).replace('nan', big).replace('NaN', big).replace('NAN', big))
                        bounds.append(tpl)
                    model["bounds"] = bounds
                elif line[:8] == "StepSize":
                    step_raw = line[9:-1].split(", ")
                    step = []
                    for stp in step_raw:
                        step.append(float(stp))
                    model["stepSize"] = step
                elif line[:8] == "InitVals":
                    init_raw = line[9:-1].split(", ")
                    init = []
                    for nt in init_raw:
                        init.append(float(nt))
                    model["initVals"] = init
                elif line[:4] == "Func": # Function
                    func = line[10:]
                    arg = args.split(", ")
                    arg_fmt = args_fmt.split(", ")
                    if arg_fmt[0] == '': # Remove empty entry
                        del arg_fmt[0]
                    func_fmt = func
                    for index, param in enumerate(arg):
                        func_fmt = func_fmt.replace(param, arg_fmt[index])
                    model["func"] = func
                    model["func_fmt"] = func_fmt
                    #print(func)
                elif line[0] == "/":
                    #print(name)
                    MODELS[name] = model
                    model = dict()
                    #i = 0
                #i += 1

        self.MODELS = MODELS

    def get_str(self, Model:str , fit_num:int , use_linear:bool = True) -> str:
        #Create python function as a string for variable parameters
        str_func_args = str() # Set empty str
        str_body = str() # Set empty str
        if use_linear:
            str_end = "slope*B+offset" # Set linear fkt
        else:
            str_end = str() # Set empty str

        model = self.MODELS[Model]
        for i in range(1, fit_num+1):   # create function/s
            str_func_args += model['args_fmt'].format(i)     #", dB{0}, R{0}, A{0}".format(i)
            if i == 1:
                str_body += model['func_fmt'].format(i)
            else:
                str_body += "+" + model['func_fmt'].format(i)
        func = "def model_fit_func(B, slope, offset{0}):\n\treturn({1} + {2})".format(str_func_args,str_body,str_end)
        return func

    def get_str_single(self, Model:str , fit_num:int , use_linear:bool = True) -> str:
        #Create python function as a string for variable parameters
        str_func_args = str() # Set empty str
        str_body = str() # Set empty str
        if use_linear:
            str_end = "slope*B+offset" # Set linear fkt
        else:
            str_end = str() # Set empty str

        model = self.MODELS[Model]
        str_func_args += model['args_fmt'].format(1)     #", dB{0}, R{0}, A{0}".format(i)
        str_body += model['func_fmt'].format(1)
        func = "def model_fit_func(B, slope, offset{0}):\n\treturn({1} + {2})".format(str_func_args,str_body,str_end)
        return func

if __name__ == '__main__':
    Fit_Models()






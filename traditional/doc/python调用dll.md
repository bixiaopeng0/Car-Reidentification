# DLL生成

1. 选择win32控制台项目
2. 选择dll项目
3. 除了空项目外，导出符号，预编译头文件，安全周期检查都要选择。
4. 如果预编译头文件出问题，在属性-c++-预编译头选型里不使用预编译头文件

按照以上步骤走就可以生成成功了，vs也会生成模板，可以按照提示做

# 调用DLL

首先抛出今天遇到的问题

1. ```c++
   AttributeError: function 'recog_color' not found
   ```

2. ```c++
   ctypes.ArgumentError: argument 7: <class 'TypeError'>: Don't know how to convert parameter 7
   ```

第一个问题是建立的c++工程，在导出头文件前面加上

```c++
#ifdef COLOR_RECOG_DLL_EXPORTS
#define COLOR_RECOG_DLL_API  extern "C" __declspec(dllexport)
#else
#define COLOR_RECOG_DLL_API  extern "C" __declspec(dllimport)
#endif
```

这一段除了extern "C"是系统自己生成的，在前面加上extern "c"就行

第二个问题是参数问题

- 第一个出问题是因为参数不匹配，传入的参数是float型，实际参数是int型
- Mat类型不可以直接传，需要变化一下，具体代码在github里面






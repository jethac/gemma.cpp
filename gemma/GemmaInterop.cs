using System;
using System.Runtime.InteropServices;
using System.Text;

namespace GemmaCpp
{
    public class GemmaException : Exception
    {
        public GemmaException(string message) : base(message) { }
    }

    public class Gemma : IDisposable
    {
        private IntPtr _context;
        private bool _disposed;

        // Optional: Allow setting DLL path
        public static string DllPath { get; set; } = "gemma.dll";

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr LoadLibrary(string lpFileName);

        static Gemma()
        {
            // Load DLL from specified path
            if (LoadLibrary(DllPath) == IntPtr.Zero)
            {
                throw new DllNotFoundException($"Failed to load {DllPath}. Error: {Marshal.GetLastWin32Error()}");
            }
        }

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr GemmaCreate(
            [MarshalAs(UnmanagedType.LPStr)] string tokenizerPath,
            [MarshalAs(UnmanagedType.LPStr)] string modelType,
            [MarshalAs(UnmanagedType.LPStr)] string weightsPath,
            [MarshalAs(UnmanagedType.LPStr)] string weightType,
            int maxLength);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GemmaDestroy(IntPtr context);

        // Delegate type for token callbacks
        public delegate bool TokenCallback(string token);

        // Keep delegate alive for duration of calls
        private GCHandle _callbackHandle;

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern int GemmaGenerate(
            IntPtr context,
            [MarshalAs(UnmanagedType.LPStr)] string prompt,
            [MarshalAs(UnmanagedType.LPStr)] StringBuilder output,
            int maxLength,
            GemmaTokenCallback callback,
            IntPtr userData);


        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern int GemmaCountTokens(
            IntPtr context,
            [MarshalAs(UnmanagedType.LPStr)] string text);

        // Native callback delegate type
        private delegate bool GemmaTokenCallback(
            [MarshalAs(UnmanagedType.LPStr)] string text,
            IntPtr userData);

        public Gemma(string tokenizerPath, string modelType, string weightsPath, string weightType, int maxLength = 8192)
        {
            _context = GemmaCreate(tokenizerPath, modelType, weightsPath, weightType, maxLength);
            if (_context == IntPtr.Zero)
            {
                throw new GemmaException("Failed to create Gemma context");
            }
        }

        public int CountTokens(string prompt)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");
            int count = GemmaCountTokens(_context, prompt);
            return count;
        }

        public string Generate(string prompt, int maxLength = 4096)
        {
            return Generate(prompt, null, maxLength);
        }

        public string Generate(string prompt, TokenCallback callback, int maxLength = 4096)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            var output = new StringBuilder(maxLength);
            GemmaTokenCallback nativeCallback = null;

            if (callback != null)
            {
                nativeCallback = (text, _) => callback(text);
                _callbackHandle = GCHandle.Alloc(nativeCallback);
            }

            try
            {
                int length = GemmaGenerate(_context, prompt, output, maxLength,
                    nativeCallback, IntPtr.Zero);

                if (length < 0)
                    throw new GemmaException("Generation failed");

                return output.ToString();
            }
            finally
            {
                if (_callbackHandle.IsAllocated)
                    _callbackHandle.Free();
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_context != IntPtr.Zero)
                {
                    GemmaDestroy(_context);
                    _context = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~Gemma()
        {
            Dispose();
        }
    }
}
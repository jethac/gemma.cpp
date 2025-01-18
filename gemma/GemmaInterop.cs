using System;
using System.Runtime.InteropServices;

namespace GemmaCpp
{
    public class Gemma : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        public Gemma(string modelPath)
        {
            _handle = GemmaCreate(modelPath);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create Gemma instance");
        }

        public string Generate(string prompt)
        {
            const int MaxLength = 8192;
            var buffer = new byte[MaxLength];

            int length = GemmaGenerate(_handle, prompt, buffer, MaxLength);
            if (length < 0)
                throw new InvalidOperationException("Generation failed");

            return System.Text.Encoding.UTF8.GetString(buffer, 0, length);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    GemmaDestroy(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~Gemma()
        {
            Dispose(false);
        }

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr GemmaCreate(
            [MarshalAs(UnmanagedType.LPStr)] string tokenizerPath,
            [MarshalAs(UnmanagedType.LPStr)] string modelType,
            [MarshalAs(UnmanagedType.LPStr)] string weightsPath,
            [MarshalAs(UnmanagedType.LPStr)] string weightType);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GemmaDestroy(IntPtr context);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern int GemmaGenerate(
            IntPtr context,
            [MarshalAs(UnmanagedType.LPStr)] string prompt,
            [MarshalAs(UnmanagedType.LPStr)] StringBuilder output,
            int maxLength);
    }
}
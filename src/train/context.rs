/// Represents a supported OpenCL platform.
pub type Platform = opencl3::platform::Platform;
/// Represents a raw device handle/ID.
pub type RawDeviceHandle = opencl3::device::cl_device_id;
/// Represents any available device via an ID that has an OpenCL ICD loader.
pub type Device = opencl3::device::Device;
/// Represents the native underlying context OpenCL uses.
pub type NativeContext = opencl3::context::Context;
/// Represents the command queue that OpenCL uses to dispatch tasks to the GPU.
pub type CommandQueue = opencl3::command_queue::CommandQueue;

/// A wrapper around a native context.
pub struct Context
{
    _platforms: Vec<Platform>,
    _raw_devices: Vec<RawDeviceHandle>,
    _devices: Vec<Device>,
    _native_ctx: NativeContext,
}

/// Represents a type of OpenCL device (e.g., `opencl3::device::CL_DEVICE_TYPE_GPU`).
pub type DeviceType = opencl3::device::cl_device_type;

impl Context
{
    /// Tries to create a new context on the first available GPU.
    pub fn new() -> Result<Self, String>
    {
        return Self::with_device_type(opencl3::device::CL_DEVICE_TYPE_GPU);
    }

    /// Tries to create a new context on the first available device of the specified type. Passing
    /// `opencl3::device::CL_DEVICE_TYPE_ALL` allows CPU OpenCL runtimes to be used as well.
    pub fn with_device_type(device_type: DeviceType) -> Result<Self, String>
    {
        use opencl3::platform::{ get_platforms };
        use opencl3::device::{ get_all_devices };

        let platforms = get_platforms().map_err(|e| {
            return format!("Failed to retrieve OpenCL platforms due to error: {e:?}.");
        })?;
        if platforms.is_empty() { return Err(format!("No OpenCL platforms supported?")); }

        let raw_devices = get_all_devices(device_type).map_err(|e| {
            return format!("Failed to retrieve OpenCL devices due to error: {e:?}.");
        })?;
        if raw_devices.is_empty() { return Err(format!("No OpenCL devices available.")); }

        let devices: Vec<Device> = raw_devices.iter().map(|x| Device::new(x.clone())).collect();

        let context = NativeContext::from_device(devices.first().unwrap()).map_err(|e| {
            return format!("Failed to create OpenCL context due to error: {e:?}.");
        })?;

        return Ok( Self {
            _platforms: platforms,
            _raw_devices: raw_devices,
            _devices: devices,
            _native_ctx: context,
        });
    }

    /// Returns the supported OpenCL platforms. The first platform is used by default.
    pub fn platforms(&self) -> &[Platform] {
        return self._platforms.as_slice();
    }
    /// Returns a slice to the raw device handles associated with this context.
    pub fn raw_devices(&self) -> &[RawDeviceHandle] {
        return self._raw_devices.as_slice();
    }
    /// Returns a slice to the (wrapped) device IDs associated with this context. The first device is used by default.
    pub fn devices(&self) -> &[Device] {
        return self._devices.as_slice();
    }
    /// Returns the device ID used or preferred by OpenCL if others exist.
    pub fn get_preferred_device(&self) -> &Device {
        return self._devices.first().unwrap();
    }
    /// Returns tne native context used by OpenCL.
    pub fn native_context(&self) -> &NativeContext {
        return &self._native_ctx;
    }


    /// Tries to create a new OpenCL command queue which will be associated with this context.
    pub fn create_command_queue(&self) -> Result<CommandQueue, String> {
        use opencl3::command_queue::{ CL_QUEUE_PROFILING_ENABLE };

        return Ok(CommandQueue::create_default_with_properties(&self._native_ctx, CL_QUEUE_PROFILING_ENABLE, 0)
            .map_err(|e| { return format!("Failed to create OpenCL command queue due to error: {e:?}."); })?);
    }

}

impl PartialEq for Context
{
    fn eq(&self, other: &Self) -> bool {
        return self._native_ctx.get() == other._native_ctx.get();
    }
}

impl Eq for Context {}

#[repr(u64)]
pub enum BufferUsage
{
    /// The kernel can read from the buffer, but cannot write to it.
    ReadOnly = opencl3::memory::CL_MEM_READ_ONLY,
    /// The kernel can write to the buffer, but cannot read from it.
    WriteOnly = opencl3::memory::CL_MEM_WRITE_ONLY,
    /// The kernel can both read from and write to the buffer.
    ReadWrite = opencl3::memory::CL_MEM_READ_WRITE,
    /// The host can write to the buffer, but cannot read from it.
    HostWriteOnly = opencl3::memory::CL_MEM_HOST_WRITE_ONLY,
    /// The host can read from the buffer, but cannot write to it.
    HostReadOnly = opencl3::memory::CL_MEM_HOST_READ_ONLY,
    /// The host does not access the buffer memory directly.
    HostNoAccess = opencl3::memory::CL_MEM_HOST_NO_ACCESS,
    /// Allocate the buffer in host-accessible memory.
    AllocHostPtr = opencl3::memory::CL_MEM_ALLOC_HOST_PTR,
    /// Use the provided host pointer as the backing storage for the buffer.
    UseHostPtr = opencl3::memory::CL_MEM_USE_HOST_PTR,
    /// Copy the provided host memory into the buffer when the buffer is created.
    CopyHostPtr = opencl3::memory::CL_MEM_COPY_HOST_PTR,
}

use opencl3::memory::{ClMem as RawBuffer, cl_mem as RawBufHandle, Buffer as NativeBuffer};

/// A wrapper around an OpenCL Buffer.
pub struct Buffer<T>
{
    _context: std::sync::Arc<Context>,
    _buffer: NativeBuffer<T>,
    _len: usize,
}

impl<T: Clone> Buffer<T>
{
    /// Tries to create an empty buffer of the specified length.
    pub fn create_empty(context: std::sync::Arc<Context>, usage: u64, len: usize) -> Result<Self, String>
    {
unsafe {
        let buf: NativeBuffer<T> = NativeBuffer::create(
            context.native_context(),
            usage,
            len,
            std::ptr::null_mut(),
        )
            .map_err(|e| format!("Failed to create buffer due to error {e:?}."))?;

        return Ok(Self {
            _context: context,
            _buffer: buf,
            _len: len,
        });
}
    }

    /// Tries to create a buffer filled with the specified data. It copies the data located at `data`.
    pub fn create(context: std::sync::Arc<Context>, usage: u64, data: &[T]) -> Result<Self, String>
    {
        use core::ffi::c_void;
unsafe {
        let buf: NativeBuffer<T> = NativeBuffer::create(
            context.native_context(),
            usage | BufferUsage::CopyHostPtr as u64,
            data.len(),
            data.as_ptr() as *mut c_void,
        )
            .map_err(|e| format!("Failed to create buffer due to error {e:?}."))?;

        return Ok(Self {
            _context: context,
            _buffer: buf,
            _len: data.len(),
        });
}
    }

    /// Returns the context that this buffer is associated with.
    pub fn context(&self) -> &std::sync::Arc<Context> {
        return &self._context;
    }
    /// Returns the raw OpenCL handle to this buffer.
    pub fn get_raw(&self) -> RawBufHandle {
        return self._buffer.get();
    }
    /// Returns a reference to the native buffer object used by OpenCL.
    pub fn get_native(&self) -> &NativeBuffer<T> {
        return &self._buffer;
    }
    /// Returns a reference to the native buffer object used by OpenCL.
    pub fn get_native_mut(&mut self) -> &mut NativeBuffer<T> {
        return &mut self._buffer;
    }
    /// Returns the length of this buffer.
    pub fn len(&self) -> usize {
        return self._len;
    }
}

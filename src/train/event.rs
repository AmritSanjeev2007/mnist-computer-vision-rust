use crate::train::context::{ Buffer, Context, CommandQueue };

use opencl3::command_queue::CL_NON_BLOCKING;
use opencl3::event::{ Event, cl_event as RawEvent };
use opencl3::kernel::ExecuteKernel;

// use core::ffi::c_void;

/// Event queues are extremely potent optimization objects. Event queues allow the user to add any
/// operation to a queue that will be immedietly executed after the previous operation has been
/// completed.
/// 
/// CPU-GPU inter-operations such as writing to, reading from and copying data are relatively
/// slow compared to operations that occur solely within the CPU or GPU. Creating an event queue
/// that literally says;
/// 
/// > *"After the previous operation has completed, start completing this one immedietly after."*
/// 
/// This eliminates the downtime of waiting for the CPU to get a response from the GPU whether the
/// initial task has completed before dispatching another.
/// 
pub struct EventQueue
{
    _context: std::sync::Arc<Context>,
    _command_queue: CommandQueue,
    // The event is kept alive (owned) until the next operation replaces it, since dropping an `Event`
    // releases it.
    _last_event: Option<Event>,
}

impl EventQueue
{
    /// Creates a new (empty) event queue associated with the specified context.
    pub fn new(context: std::sync::Arc<Context>) -> Result<Self, String>
    {  
        let cmd_queue = context.create_command_queue()?;

        return Ok(Self {
            _context: context,
            _command_queue: cmd_queue,
            _last_event: None,
        });
    }

    /// Appends a read buffer operation to the end of this event queue.
    /// 
    /// It will try to copy `data.len()` elements from the specified `offset` (in elements) from the start of this
    /// buffer into the specified out `data` slice.
    /// 
    /// The read is non-blocking, `data` must not be used until `finish()` has been called.
    pub fn enqueue_read_buffer<T: Clone>(&mut self, buf: &Buffer<T>, data: &mut [T], offset: usize) -> Result<&mut Self, String>
    {
        if offset + data.len() > buf.len() {
            return Err(format!("Size of offset + data.len is greater than the length of the specified buffer."));
        }

        let prev_event_handle: Option<RawEvent> = self._last_event.as_ref().map(|ev| ev.get());
        let prev_event: &[RawEvent] = prev_event_handle.as_ref().map_or(&[], |ev| std::slice::from_ref(ev));

unsafe {
        let event = self._command_queue.enqueue_read_buffer(
            buf.get_native(),
            CL_NON_BLOCKING as u32,
            offset * std::mem::size_of::<T>(),
            data,
            prev_event,
        ).map_err(|e| format!("Failed to enqueue_read_buffer due to error: {e:?}."))?;

        self._last_event = Some(event);
}
        return Ok(self);
    }

    /// Appends a write buffer operation to the end of this event queue. It will try to copy `data` into this buffer
    /// starting at the specified `offset` (in elements).
    /// 
    /// The write is non-blocking, `data` must outlive the write i.e., until `finish()` has been called.
    pub fn enqueue_write_buffer<T: Clone>(&mut self, buf: &mut Buffer<T>, data: &[T], offset: usize) -> Result<&mut Self, String>
    {
        if offset + data.len() > buf.len() {
           return Err(format!("Size of offset + data.len is greater than the length of the specified buffer."));
        }

        let prev_event_handle: Option<RawEvent> = self._last_event.as_ref().map(|ev| ev.get());
        let prev_event: &[RawEvent] = prev_event_handle.as_ref().map_or(&[], |ev| std::slice::from_ref(ev));
        
unsafe {
        let event = self._command_queue.enqueue_write_buffer(
            buf.get_native_mut(),
            CL_NON_BLOCKING as u32,
            offset * std::mem::size_of::<T>(),
            data,
            prev_event,
        ).map_err(|e| format!("Failed to enqueue_write_buffer due to error: {e:?}."))?;

        self._last_event = Some(event);
}
        return Ok(self);
    }

    /// Appends a kernel execution to the end of this event queue. The kernel arguments and work sizes must
    /// already be set on `kernel`.
    pub fn enqueue_kernel(&mut self, kernel: &mut ExecuteKernel) -> Result<&mut Self, String>
    {
        if let Some(prev_event) = self._last_event.as_ref() {
            kernel.set_wait_event(prev_event);
        }

unsafe {
        let event = kernel.enqueue_nd_range(&self._command_queue)
            .map_err(|e| format!("Failed to enqueue kernel due to error: {e:?}."))?;

        self._last_event = Some(event);
}
        return Ok(self);
    }

    /// Blocks the calling thread until all enqueued events have completed. This function allows for
    /// synchronization between the CPU-GPU.
    pub fn finish(&self) -> Result<(), String>
    {
        return self._command_queue
            .finish()
            .map_err(|e| format!("Failed to finish queue due to error: {e:?}."));
    }
}

import * as React from "react"
import { cn } from "../../lib/utils"

const Progress = React.forwardRef(({ className, value, ...props }, ref) => (
    <div
        ref={ref}
        className={cn(
            "relative h-2 w-full overflow-hidden rounded-full bg-slate-800",
            className
        )}
        {...props}
    >
        <div
            className="h-full w-full flex-1 bg-indigo-500 transition-all duration-500 ease-in-out"
            style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
        />
    </div>
))
Progress.displayName = "Progress"

export { Progress }

"use client";

import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Box, PathData, Point } from "@/types/path";
import { addPathData } from "@/actions/db";

export default function Page() {
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [currentBox, setCurrentBox] = useState<string | null>(null);
  const [mousePath, setMousePath] = useState<Point[]>([]);
  const [isTracking, setIsTracking] = useState(false);
  const [pathData, setPathData] = useState<PathData | null>(null);
  const [isMobile, setIsMobile] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const checkIfMobile = () => {
      setIsMobile(
        window.innerWidth <= 768 ||
          /iPhone|iPad|iPod|Android/i.test(navigator.userAgent)
      );
    };

    checkIfMobile();
    window.addEventListener("resize", checkIfMobile);
    generateBoxes();

    return () => window.removeEventListener("resize", checkIfMobile);
  }, []);

  const generateBoxes = () => {
    if (containerRef.current) {
      const { width, height } = containerRef.current.getBoundingClientRect();
      const newBoxes: Box[] = [
        {
          id: "A",
          x: Math.random() * (width - 60),
          y: Math.random() * (height - 60),
        },
        {
          id: "B",
          x: Math.random() * (width - 60),
          y: Math.random() * (height - 60),
        },
      ];
      setBoxes(newBoxes);
      setCurrentBox("A");
      setMousePath([]);
      setIsTracking(false);
      setPathData(null);
    }
  };

  const handleBoxClick = (boxId: string) => {
    if (boxId === currentBox) {
      if (boxId === "A") {
        setCurrentBox("B");
        setIsTracking(true);
      } else {
        setIsTracking(false);
        const newPathData: PathData = {
          start: mousePath[0],
          end: mousePath[mousePath.length - 1],
          path: mousePath,
        };
        setPathData(newPathData);
        console.log("Path data:", newPathData);

        addPathData(newPathData);

        setTimeout(generateBoxes, 100); // Reset after delay
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (isTracking && containerRef.current) {
      const { left, top } = containerRef.current.getBoundingClientRect();
      setMousePath((prev) => [
        ...prev,
        {
          x: e.clientX - left,
          y: e.clientY - top,
          timestamp: Date.now(),
        },
      ]);
    }
  };

  if (isMobile) {
    return (
      <div className="flex items-center justify-center min-h-screen w-full bg-gray-100 p-4">
        <h1 className="text-xl font-bold text-red-500">
          This application is only intended for desktop use with a mouse or
          trackpad.
        </h1>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen w-full bg-gray-100 p-4">
      <h1 className="text-2xl font-bold mb-4">Project Scribe</h1>
      <div
        ref={containerRef}
        className="relative w-full h-[calc(100vh-8rem)] bg-white border-2 border-gray-300 rounded-lg overflow-hidden"
        onMouseMove={handleMouseMove}
      >
        {boxes.map((box) => (
          <div
            key={box.id}
            className={`absolute w-6 h-6 rounded-full flex items-center justify-center text-white cursor-pointer ${
              box.id === "A" ? "bg-blue-500" : "bg-green-500"
            } ${
              currentBox === box.id ? "ring-2 ring-offset-2 ring-black" : ""
            }`}
            style={{ left: `${box.x}px`, top: `${box.y}px` }}
            onClick={() => handleBoxClick(box.id)}
          >
            {box.id}
          </div>
        ))}
        {isTracking && (
          <svg className="absolute top-0 left-0 w-full h-full pointer-events-none">
            <path
              d={mousePath
                .map(
                  (point, index) =>
                    `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`
                )
                .join(" ")}
              fill="none"
              stroke="red"
              strokeWidth="2"
            />
          </svg>
        )}
      </div>
      <Button onClick={generateBoxes} className="mt-4">
        Reset
      </Button>
      <div className="mt-4 text-sm text-gray-600" aria-live="polite">
        {currentBox === "A" && "Click point A to start tracking"}
        {currentBox === "B" && "Now click point B to finish"}
        {!currentBox && pathData && "Path recorded! Check the console for data"}
      </div>
    </div>
  );
}

import { useEffect, useRef } from 'react';
import React from 'react';

type DigitImageProps = {
  data: number[];
  label: number;
  scale: number;
  onClick: () => void;
  isSelected: boolean;
};

const DigitImage: React.FC<DigitImageProps> = ({ data, label, scale, onClick, isSelected}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
  
    useEffect(() => {
      if (canvasRef.current) { // Fill Canvas with MNIST pixels
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) {
          let imgData = ctx.createImageData(28 * scale, 28 * scale);
          for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
              const pixel = data[y * 28 + x];
              for (let dy = 0; dy < scale; dy++) {
                for (let dx = 0; dx < scale; dx++) {
                  const index = ((y * scale + dy) * 28 * scale + (x * scale + dx)) * 4;
                  imgData.data[index] = pixel;      // Red
                  imgData.data[index + 1] = pixel;  // Green
                  imgData.data[index + 2] = pixel;  // Blue
                  imgData.data[index + 3] = 255;    // Alpha
                }
              }
            }
          }
          ctx.putImageData(imgData, 0, 0);
        }
      }
    }, [data, scale]);

    const borderColor = isSelected ? '#54b3d6' : 'transparent';
  
    return (
      <div onClick={onClick} style={{ cursor: 'pointer', border: `5px solid ${borderColor}`, height: 28*scale}} >
        <canvas ref={canvasRef} width={28 * scale} height={28 * scale} />
      </div>
    );
  };
  
  export default DigitImage;
  
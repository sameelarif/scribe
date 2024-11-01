import axios from "axios";
import { KeyInput, Page } from "puppeteer";
import { setTimeout } from "timers/promises";

interface Point {
  x: number;
  y: number;
}

interface MoveMouseOptions {
  visualize?: boolean;
  log?: boolean;
}

interface ScribeOptions {
  hesitationDelay: number;
  clickDelay: number;
  moveDelay: number;
  apiUrl: string;
  visualize: boolean;
  startPosition: Point;
}

interface TypeOptions {
  wpm?: number;
}

export default class Scribe {
  page: Page;
  options: ScribeOptions;
  mouse: Point;

  constructor(page: Page, options: Partial<ScribeOptions> = {}) {
    this.page = page;

    const {
      hesitationDelay = 15,
      clickDelay = 50,
      moveDelay = 0,
      apiUrl = "http://localhost:3000/generate_path",
      visualize = false,
      startPosition = { x: 0, y: 0 },
    } = options;

    this.options = {
      hesitationDelay,
      clickDelay,
      moveDelay,
      apiUrl,
      visualize,
      startPosition,
    };

    if (visualize) {
      this.installMouseHelper();
    }

    this.mouse = startPosition;
    this.installMouseTracker();
  }

  public async moveMouse(
    startPoint: Point,
    endPoint: Point,
    options: MoveMouseOptions = {}
  ): Promise<void> {
    const { visualize = false, log = false } = options;

    try {
      const response = await axios.get(this.options.apiUrl, {
        params: {
          start_point: [startPoint.x, startPoint.y].join(","),
          end_point: [endPoint.x, endPoint.y].join(","),
          visualize: false, // We don't need the plot image
        },
      });

      if (response.status !== 200 || !response.data.path) {
        throw new Error("Failed to retrieve path from API");
      }

      const path: [number, number][] = this.interpolatePath(response.data.path);

      if (log) {
        console.log("Starting mouse movement along path");
      }

      await this.page.mouse.move(path[0][0], path[0][1]);
      if (log) {
        console.log(
          `Moved mouse to initial position: (${path[0][0]}, ${path[0][1]})`
        );
      }

      for (let i = 0; i < path.length - 1; i++) {
        const start = path[i];
        const end = path[i + 1];

        // Calculate distance between points
        const distance = Math.hypot(end[0] - start[0], end[1] - start[1]);
        const steps = Math.max(1, Math.floor(distance / 2));

        await this.page.mouse.move(end[0], end[1], { steps });

        if (log) {
          console.log(`Moved mouse to position: (${end[0]}, ${end[1]})`);
        }

        if (this.options.moveDelay > 0)
          await setTimeout(this.options.moveDelay);
      }

      if (log) {
        console.log("Completed mouse movement along path");
      }
    } catch (error) {
      console.error("Error in moveMouse:", error);
      throw error;
    }
  }

  public async click(selector: string): Promise<void> {
    const targetEl = await this.page.$(selector);

    if (!targetEl) {
      throw new Error(`Unable to locate element for selector: \`${selector}\``);
    }

    const targetBoundingBox = await targetEl.boundingBox();

    if (!targetBoundingBox) {
      throw new Error("No bounding box for target element");
    }

    await this.moveMouse(this.mouse, {
      x:
        targetBoundingBox.x +
        Math.floor(Math.random() * targetBoundingBox.width),
      y:
        targetBoundingBox.y +
        Math.floor(Math.random() * targetBoundingBox.height),
    });

    await this.page.mouse.down();
    await setTimeout(this.options.clickDelay);
    await this.page.mouse.up();
  }

  public async type(text: string, options: TypeOptions = {}) {
    // Set WPM between 60-100 if not provided
    if (!options.wpm) {
      options.wpm = 60 + Math.floor(Math.random() * 40);
    }

    const charactersPerMinute = options.wpm * 5;
    const delay = 60000 / charactersPerMinute;
    const dwellTime = delay / 2;

    const deviation = 0.2;

    for (let i = 0; i < text.length; i++) {
      const c = text.charAt(i) as KeyInput;

      await this.page.keyboard.down(c);
      await setTimeout(
        dwellTime + Math.random() * deviation * (Math.random() > 0.5 ? 1 : -1)
      );
      await this.page.keyboard.up(c);

      await setTimeout(
        delay -
          dwellTime +
          Math.random() * deviation * (Math.random() > 0.5 ? 1 : -1)
      );
    }
  }

  private interpolatePath(path: [number, number][]): [number, number][] {
    const interpolatedPath: [number, number][] = [];
    const totalPoints = path.length;

    for (let i = 0; i < totalPoints - 1; i++) {
      const [x1, y1] = path[i];
      const [x2, y2] = path[i + 1];

      interpolatedPath.push([x1, y1]);

      // Determine the number of intermediate points
      const distance = Math.hypot(x2 - x1, y2 - y1);
      const numIntermediatePoints = Math.max(1, Math.floor(distance / 10));

      for (let j = 1; j < numIntermediatePoints; j++) {
        const t = j / numIntermediatePoints;
        const x = x1 + (x2 - x1) * t;
        const y = y1 + (y2 - y1) * t;
        interpolatedPath.push([x, y]);
      }
    }

    // Add the last point
    interpolatedPath.push(path[totalPoints - 1]);

    return interpolatedPath;
  }

  private async installMouseHelper(): Promise<void> {
    await this.page.evaluateOnNewDocument(() => {
      const attachListener = (): void => {
        const box = document.createElement("p-mouse-pointer");
        const styleElement = document.createElement("style");

        styleElement.innerHTML = `
            p-mouse-pointer {
              pointer-events: none;
              position: absolute;
              top: 0;
              z-index: 10000;
              left: 0;
              width: 20px;
              height: 20px;
              background: rgba(0,0,0,.4);
              border: 1px solid white;
              border-radius: 10px;
              box-sizing: border-box;
              margin: -10px 0 0 -10px;
              padding: 0;
              transition: background .2s, border-radius .2s, border-color .2s;
            }
            p-mouse-pointer.button-1 {
              transition: none;
              background: rgba(0,0,0,0.9);
            }
            p-mouse-pointer.button-2 {
              transition: none;
              border-color: rgba(0,0,255,0.9);
            }
            p-mouse-pointer.button-3 {
              transition: none;
              border-radius: 4px;
            }
            p-mouse-pointer.button-4 {
              transition: none;
              border-color: rgba(255,0,0,0.9);
            }
            p-mouse-pointer.button-5 {
              transition: none;
              border-color: rgba(0,255,0,0.9);
            }
            p-mouse-pointer-hide {
              display: none
            }
          `;

        document.head.appendChild(styleElement);
        document.body.appendChild(box);

        document.addEventListener(
          "mousemove",
          (event) => {
            console.log("event");
            box.style.left = String(event.pageX) + "px";
            box.style.top = String(event.pageY) + "px";
            box.classList.remove("p-mouse-pointer-hide");
            updateButtons(event.buttons);
          },
          true
        );

        document.addEventListener(
          "mousedown",
          (event) => {
            updateButtons(event.buttons);
            box.classList.add("button-" + String(event.which));
            box.classList.remove("p-mouse-pointer-hide");
          },
          true
        );

        document.addEventListener(
          "mouseup",
          (event) => {
            updateButtons(event.buttons);
            box.classList.remove("button-" + String(event.which));
            box.classList.remove("p-mouse-pointer-hide");
          },
          true
        );

        document.addEventListener(
          "mouseleave",
          (event) => {
            updateButtons(event.buttons);
            box.classList.add("p-mouse-pointer-hide");
          },
          true
        );

        document.addEventListener(
          "mouseenter",
          (event) => {
            updateButtons(event.buttons);
            box.classList.remove("p-mouse-pointer-hide");
          },
          true
        );

        function updateButtons(buttons: number): void {
          for (let i = 0; i < 5; i++) {
            box.classList.toggle(
              "button-" + String(i),
              Boolean(buttons & (1 << i))
            );
          }
        }
      };

      if (document.readyState !== "loading") {
        attachListener();
      } else {
        window.addEventListener("DOMContentLoaded", attachListener, false);
      }
    });
  }

  private async installMouseTracker(): Promise<void> {
    await this.page.exposeFunction(
      "updateMousePosition",
      (x: number, y: number) => (this.mouse = { x, y })
    );

    await this.page.evaluateOnNewDocument(() => {
      window.addEventListener("mousemove", (event: MouseEvent) => {
        (window as any).updateMousePosition(event.pageX, event.pageY);
      });
    });
  }
}

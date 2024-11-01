import axios from "axios";
import { Page } from "puppeteer";
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
      moveDelay = 25,
      apiUrl = "http://localhost:3000/generate_path",
      visualize = false,
    } = options;

    this.options = {
      hesitationDelay,
      clickDelay,
      moveDelay,
      apiUrl,
      visualize,
    };

    if (visualize) {
      this.installMouseHelper();
    }

    this.mouse = { x: 0, y: 0 };
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

      const path: [number, number][] = response.data.path;
      if (log) {
        console.log("Starting mouse movement along path");
      }

      await this.page.mouse.move(path[0][0], path[0][1]);
      if (log) {
        console.log(
          `Moved mouse to initial position: (${path[0][0]}, ${path[0][1]})`
        );
      }

      for (let i = 1; i < path.length; i++) {
        await this.page.mouse.move(path[i][0], path[i][1]);
        if (log) {
          console.log(
            `Moved mouse to position: (${path[i][0]}, ${path[i][1]})`
          );
        }
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
